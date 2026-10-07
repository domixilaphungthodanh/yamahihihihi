"""World Cup 2022 football player distance project."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urljoin

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HUB = 'https://www.fifatrainingcentre.com/en/fwc2022/post-match-summaries/post-match-summary-reports.php'
DATAHUB = 'https://datahub.io/football/worldcup/_r/-/'
FEATURES = ['position', 'minutes_played', 'started', 'extra_time', 'team_goals', 'opponent_goals', 'team_passes', 'team_shots']
NUMERIC = FEATURES[1:]
POSITION = {'GK': 'goalkeeper', 'DF': 'defender', 'MF': 'midfielder', 'FW': 'forward'}

def source_manifest():
    html = requests.get(HUB, timeout=45).text
    soup = BeautifulSoup(html, 'html.parser')
    urls = [urljoin(HUB, a['href']).replace('http://', 'https://') for a in soup.find_all('a', href=True) if '.pdf' in a['href'].lower()]
    urls = list(dict.fromkeys(urls))
    if len(urls) != 64:
        raise ValueError(f'Expected 64 FIFA reports, found {len(urls)}')
    return urls

def download(url, path):
    if path.exists(): return path
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    if not response.content.startswith(b'%PDF'):
        raise ValueError(f'Not a PDF: {url}')
    path.write_bytes(response.content)
    return path

def parse_roster(page, side):
    words = page.extract_words()
    if side == 'left':
        pos_words = [w for w in words if 72 < w['x0'] < 95 and w['text'] in POSITION]
        number_x = (45, 72)
    else:
        pos_words = [w for w in words if 860 < w['x0'] < 888 and w['text'] in POSITION]
        number_x = (885, 920)
    result = {}
    for p in pos_words:
        near = [w for w in words if number_x[0] < w['x0'] < number_x[1] and abs(w['top'] - p['top']) < 5 and w['text'].isdigit()]
        if len(near) != 1: continue
        number = int(near[0]['text'])
        if number in result: continue
        result[number] = {'position': POSITION[p['text']], 'started': int(p['top'] < 307)}
    return result

def parse_score_stats(page):
    text = page.extract_text() or ''
    def pair(pattern):
        found = re.search(pattern, text)
        if not found: raise ValueError(f'Stat unavailable: {pattern}')
        return int(found.group(1)), int(found.group(2))
    goals = pair(r'(?m)^\s*(\d+) Goals (\d+)\s*$')
    shots = pair(r'(?m)^\s*(\d+) \(\d+\) Attempts at Goal \(On Target\) (\d+) \(\d+\)\s*$')
    passes = pair(r'(?m)^\s*(\d+) \(\d+\) Total Passes \(Complete\) (\d+) \(\d+\)\s*$')
    return [{'team_goals': goals[i], 'opponent_goals': goals[1-i], 'team_shots': shots[i], 'team_passes': passes[i]} for i in [0,1]]

def physical_page(page, path, page_index, tesseract):
    text = page.extract_text() or ''
    heading = re.search(r'Physical Data\s+([^\n]+)', text)
    if not heading: return None
    team = heading.group(1).strip()
    # Render only the two physical-data pages. OCR is needed because this PDF's digit font has no usable Unicode map.
    prefix = ROOT / 'data' / 'raw' / f'{path.stem}-p{page_index+1}'
    subprocess.run([str(tesseract['pdftoppm']), '-f', str(page_index+1), '-l', str(page_index+1), '-scale-to', '1800', '-png', '-singlefile', str(path), str(prefix)], check=True, capture_output=True)
    picture = prefix.with_suffix('.png')
    try:
        proc = subprocess.run([str(tesseract['ocr']), str(picture), 'stdout', '--psm', '6'], capture_output=True, text=True, check=True)
    finally:
        picture.unlink(missing_ok=True)
    rows = {}
    for line in proc.stdout.splitlines():
        match = re.match(r'^\s*(\d{1,2})\s+(.+?)\s+(\d{3,5}\.\d)\s+([\d.]+)', line)
        if match:
            number, name, distance = int(match[1]), match[2], float(match[3])
            zones = [float(value) for value in re.findall(r'\d+\.\d+', line[match.end(3):])]
            if len(zones) >= 5 and abs(distance - sum(zones[:5])) > max(100, .03 * distance):
                continue
            if 100 <= distance <= 20000 and number not in rows:
                rows[number] = {'player': name.strip(), 'distance_km': distance / 1000}
    return team, rows

def report_rows(path, tesseract):
    with pdfplumber.open(path) as pdf:
        cover = pdf.pages[0].extract_text() or ''
        number = re.search(r'Match\s+(\d+)', cover)
        if not number: raise ValueError('Match number missing')
        match_id = f'M-2022-{int(number[1]):02d}'
        roster = [parse_roster(pdf.pages[1], side) for side in ['left','right']]
        stats = parse_score_stats(pdf.pages[2])
        physical = []
        for i in range(max(0,len(pdf.pages)-4), len(pdf.pages)):
            text = pdf.pages[i].extract_text() or ''
            if 'Physical Data' in text:
                item = physical_page(pdf.pages[i], path, i, tesseract)
                if item: physical.append(item)
        if len(physical) != 2: raise ValueError(f'Expected 2 physical tables; got {len(physical)}')
        rows, issues = [], []
        for side, (team, distances) in enumerate(physical):
            for number, measure in distances.items():
                if number not in roster[side]:
                    issues.append({'match_id': match_id, 'team': team, 'shirt_number': number, 'reason': 'OCR row has no roster match'})
                    continue
                rows.append({'match_id': match_id, 'team': team, 'shirt_number': number, **roster[side][number], **stats[side], **measure})
            for number in roster[side].keys() - distances.keys():
                issues.append({'match_id': match_id, 'team': team, 'shirt_number': number, 'reason': 'roster row has no OCR distance (may not have played)'})
        return match_id, rows, issues

def build_data(limit=None):
    raw = ROOT / 'data' / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    processed = ROOT / 'data' / 'processed'; processed.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / 'reports'; report_dir.mkdir(exist_ok=True)
    urls = source_manifest()
    if limit: urls = urls[:limit]
    (report_dir / 'sources.json').write_text(json.dumps({'fifa_hub': HUB, 'pdf_urls': urls, 'secondary_source': DATAHUB}, indent=2), encoding='utf-8')
    def task(item):
        i,url = item
        path = raw / f'report-{i:02d}.pdf'
        download(url,path)
        return i,url,path
    paths=[]; issues=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(task,item) for item in enumerate(urls,1)]):
            try: paths.append(future.result())
            except Exception as e: issues.append({'report': 'download', 'reason': str(e)})
    tools = {
        'ocr': Path(os.environ.get('TESSERACT_EXE') or shutil.which('tesseract') or r'C:\Program Files\Tesseract-OCR\tesseract.exe'),
        'pdftoppm': Path(os.environ.get('PDFTOPPM_EXE') or shutil.which('pdftoppm') or r'C:\Users\Dell XPS\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')}
    if not all(p.exists() for p in tools.values()): raise FileNotFoundError('Tesseract or Poppler missing; see README')
    all_rows=[]; matches=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(report_rows,path,tools):(i,url) for i,url,path in paths}
        for future in as_completed(jobs):
            i,url=jobs[future]
            try:
                match_id,rows,failed=future.result()
                all_rows.extend(rows); issues.extend(failed)
                matches.append({'match_id':match_id,'url':url,'rows':len(rows)})
                print(f'{len(matches)}/{len(urls)} {match_id}: {len(rows)} players',flush=True)
            except Exception as e: issues.append({'report':url,'reason':str(e)})
    if not all_rows: raise RuntimeError('No validated player rows extracted')
    frame = pd.DataFrame(all_rows)
    frame.to_csv(raw/'ocr_rows.csv',index=False)
    return finalize_rows(frame, issues, matches)

def finalize_rows(frame, issues, matches):
    raw = ROOT / 'data' / 'raw'
    processed = ROOT / 'data' / 'processed'
    report_dir = ROOT / 'reports'
    frame=frame.copy()
    # Fjelstul World Cup Database (CC BY-SA 4.0) supplies substitution times and match extra-time flag.
    matches_df = pd.read_csv(DATAHUB + 'matches.csv').query("tournament_id == 'WC-2022'")
    subs = pd.read_csv(DATAHUB + 'substitutions.csv').query("tournament_id == 'WC-2022'")
    aliases={'USA':'United States','IR Iran':'Iran','Korea Republic':'South Korea'}
    def norm(name):
        return re.sub(r'[^a-z]','',aliases.get(name,name).lower())
    # FIFA match numbers and Fjelstul match IDs use different ordering.
    lookup={}
    for r in matches_df.itertuples():
        key=tuple(sorted([(norm(r.home_team_name),int(r.home_team_score)),(norm(r.away_team_name),int(r.away_team_score))]))
        if key in lookup: raise ValueError(f'Ambiguous Fjelstul match key: {key}')
        lookup[key]=r.match_id
    fifa_to_fjelstul={}
    for fifa_id,g in frame.groupby('match_id'):
        teams=g[['team','team_goals']].drop_duplicates()
        key=tuple(sorted([(norm(r.team),int(r.team_goals)) for r in teams.itertuples()]))
        if len(key)!=2 or key not in lookup: raise ValueError(f'Could not match FIFA report {fifa_id}: {key}')
        fifa_to_fjelstul[fifa_id]=lookup[key]
    frame['fifa_match_id']=frame.match_id
    frame['match_id']=frame.match_id.map(fifa_to_fjelstul)
    match_info = matches_df.set_index('match_id')
    frame['extra_time'] = frame.match_id.map(match_info.extra_time)
    frame['match_minutes'] = np.where(frame.extra_time == 1,120,90)
    # Match ID, team and jersey are stable identifiers across both sources.
    frame['team_key'] = frame.team.replace(aliases).str.lower().str.replace(r'[^a-z]', '', regex=True)
    subs['team_key'] = subs.team_name.str.lower().str.replace(r'[^a-z]', '', regex=True)
    subs['key'] = list(zip(subs.match_id, subs.team_key, subs.shirt_number))
    subgroups = {k: g for k,g in subs.groupby('key')}
    minutes=[]
    for row in frame.itertuples():
        group = subgroups.get((row.match_id,row.team_key,row.shirt_number))
        start = 0 if row.started else None
        end = int(row.match_minutes)
        if group is not None:
            entering = group[group.coming_on == 1]
            leaving = group[group.going_off == 1]
            if not entering.empty: start = int(entering.iloc[0].minute_regulation) + int(entering.iloc[0].minute_stoppage)
            if not leaving.empty: end = int(leaving.iloc[0].minute_regulation) + int(leaving.iloc[0].minute_stoppage)
        minutes.append(np.nan if start is None or end <= start else end-start)
    frame['minutes_played'] = minutes
    frame.to_csv(raw/'extracted_before_validation.csv',index=False)
    bad = frame[frame.minutes_played.isna() | (frame.distance_km <= 0) | (frame.distance_km > 20) | frame.extra_time.isna()]
    issues.extend([{'match_id':r.match_id,'team':r.team,'shirt_number':r.shirt_number,'reason':'invalid minutes or distance'} for r in bad.itertuples()])
    frame = frame.drop(bad.index).drop(columns=['team_key','match_minutes'])
    frame = frame.drop_duplicates(['match_id','team','shirt_number'],keep=False)
    frame.to_csv(processed/'player_matches.csv', index=False)
    pd.DataFrame(issues).to_csv(report_dir/'extraction_issues.csv', index=False)
    pd.DataFrame(matches).to_csv(report_dir/'extracted_matches.csv', index=False)
    return frame

class PositionMinutesBaseline(BaseEstimator, RegressorMixin):
    def fit(self,X,y):
        t = pd.DataFrame({'position':X['position'].to_numpy(),'rate':np.asarray(y)/X['minutes_played'].to_numpy()})
        self.rates_ = t.groupby('position').rate.median().to_dict()
        self.default_ = float(t.rate.median())
        return self
    def predict(self,X):
        return X['minutes_played'].to_numpy() * X['position'].map(self.rates_).fillna(self.default_).to_numpy()

def make_model(estimator, scale=False):
    steps=[('impute',SimpleImputer(strategy='median'))]
    if scale: steps.append(('scale',StandardScaler()))
    from sklearn.pipeline import Pipeline as InnerPipeline
    pre=ColumnTransformer([('num',InnerPipeline(steps),NUMERIC),('position',OneHotEncoder(handle_unknown='ignore'),['position'])])
    return Pipeline([('preprocess',pre),('model',estimator)])

def train():
    df=pd.read_csv(ROOT/'data'/'processed'/'player_matches.csv')
    if df.match_id.nunique()<12 or set(df.position)!=set(POSITION.values()): raise ValueError('Insufficient validated matches or positions')
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    train_idx,test_idx=next(splitter.split(df,groups=df.match_id))
    train_df,test_df=df.iloc[train_idx],df.iloc[test_idx]
    X_train,y_train=train_df[FEATURES],train_df.distance_km
    X_test,y_test=test_df[FEATURES],test_df.distance_km
    cv=GroupKFold(n_splits=5)
    candidates={
        'position_minutes_baseline':(PositionMinutesBaseline(),{}),
        'ridge':(make_model(Ridge(),True),{'model__alpha':[.1,1,10,100]}),
        'random_forest':(make_model(RandomForestRegressor(n_estimators=250,random_state=42,n_jobs=-1)),{'model__max_depth':[None,8],'model__min_samples_leaf':[1,3,6]})}
    results=[]; fitted={}
    for name,(model,params) in candidates.items():
        search=GridSearchCV(model,params,scoring='neg_mean_absolute_error',cv=cv,n_jobs=1)
        search.fit(X_train,y_train,groups=train_df.match_id)
        fitted[name]=search.best_estimator_
        results.append({'model':name,'cv_mae_km':-search.best_score_,'params':str(search.best_params_)})
    ranking=pd.DataFrame(results).sort_values('cv_mae_km').reset_index(drop=True)
    name=ranking.iloc[0]['model']; model=fitted[name]
    predictions=model.predict(X_test)
    baseline=fitted['position_minutes_baseline'].predict(X_test)
    metrics={'model':name,'mae_km':float(mean_absolute_error(y_test,predictions)),'rmse_km':float(np.sqrt(mean_squared_error(y_test,predictions))),'r2':float(r2_score(y_test,predictions)),'baseline_test_mae_km':float(mean_absolute_error(y_test,baseline)),'train_matches':int(train_df.match_id.nunique()),'test_matches':int(test_df.match_id.nunique()),'train_rows':len(train_df),'test_rows':len(test_df)}
    per_position=pd.DataFrame({'position':test_df.position,'error':abs(y_test.to_numpy()-predictions)}).groupby('position').error.agg(['count','mean']).reset_index().rename(columns={'count':'n','mean':'mae_km'})
    for folder in ['models','reports']:(ROOT/folder).mkdir(exist_ok=True)
    joblib.dump(model,ROOT/'models'/'distance_pipeline.joblib')
    metadata={'metrics':metrics,'features':FEATURES,'ranges':{c:[float(train_df[c].min()),float(train_df[c].max())] for c in NUMERIC},'positions':list(POSITION.values())}
    (ROOT/'models'/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    ranking.to_csv(ROOT/'reports'/'model_comparison.csv',index=False)
    per_position.to_csv(ROOT/'reports'/'position_metrics.csv',index=False)
    errors=test_df[['match_id','team','player','position','minutes_played','distance_km']].copy()
    errors['predicted_km']=predictions;errors['absolute_error_km']=abs(y_test.to_numpy()-predictions)
    errors.sort_values('absolute_error_km',ascending=False).to_csv(ROOT/'reports'/'test_predictions.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(y_test,predictions,alpha=.6); lo=min(y_test.min(),min(predictions)); hi=max(y_test.max(),max(predictions)); axes[0].plot([lo,hi],[lo,hi],'r--'); axes[0].set(xlabel='Actual km',ylabel='Predicted km')
    axes[1].bar(per_position.position,per_position.mae_km);axes[1].set(ylabel='MAE (km)',title='Error by position');fig.tight_layout();fig.savefig(ROOT/'reports'/'evaluation.png',dpi=150);plt.close(fig)
    (ROOT/'reports'/'results.md').write_text('# Kết quả\n\n'+json.dumps(metrics,indent=2)+'\n\n'+per_position.to_string(index=False)+'\n\nChỉ đánh giá trên các trận thuộc World Cup 2022. Số phút là ước tính từ thời điểm thay người và thời lượng trận danh nghĩa.\n',encoding='utf-8')
    return metrics

def predict(payload):
    if not isinstance(payload,dict): raise ValueError('JSON object required')
    missing=set(FEATURES)-payload.keys()
    if missing: raise ValueError(f'Missing fields: {sorted(missing)}')
    if payload['position'] not in POSITION.values(): raise ValueError('Invalid position')
    for field in NUMERIC:
        value=payload[field]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0: raise ValueError(f'Invalid {field}')
    if payload['minutes_played']<=0 or payload['minutes_played']>130: raise ValueError('Invalid minutes_played')
    frame=pd.DataFrame([payload],columns=FEATURES)
    metadata=json.loads((ROOT/'models'/'metadata.json').read_text(encoding='utf-8'))
    model=joblib.load(ROOT/'models'/'distance_pipeline.joblib')
    warnings=[f'{c} outside training range' for c,(lo,hi) in metadata['ranges'].items() if not lo<=payload[c]<=hi]
    return {'distance_km':round(float(model.predict(frame)[0]),3),'model':metadata['metrics']['model'],'warnings':warnings}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','finalize','train','predict']);parser.add_argument('--input',type=Path);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.command=='prepare': print(build_data(args.limit).shape)
    elif args.command=='finalize':
        frame=pd.read_csv(ROOT/'data'/'raw'/'ocr_rows.csv')
        issues=pd.read_csv(ROOT/'reports'/'extraction_issues.csv').query("reason != 'invalid minutes or distance'").to_dict('records')
        matches=pd.read_csv(ROOT/'reports'/'extracted_matches.csv').to_dict('records')
        print(finalize_rows(frame,issues,matches).shape)
    elif args.command=='train': print(json.dumps(train(),indent=2))
    else:
        if not args.input: parser.error('predict needs --input JSON file')
        print(json.dumps(predict(json.loads(args.input.read_text(encoding='utf-8'))),indent=2))

if __name__=='__main__': main()
