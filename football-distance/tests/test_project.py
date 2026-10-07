import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import project

class ProjectTests(unittest.TestCase):
    def test_position_mapping_and_baseline(self):
        self.assertEqual(set(project.POSITION),{'GK','DF','MF','FW'})
        X=pd.DataFrame({'position':['goalkeeper','goalkeeper','forward','forward'],'minutes_played':[90,45,90,45]})
        y=np.array([4.5,2.25,9.0,4.5])
        model=project.PositionMinutesBaseline().fit(X,y)
        np.testing.assert_allclose(model.predict(X),y)

    def test_match_group_separation(self):
        df=pd.DataFrame({'match_id':[f'm{i}' for i in range(12) for _ in range(3)]})
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,groups=df.match_id))
        self.assertFalse(set(df.iloc[a].match_id)&set(df.iloc[b].match_id))

    def test_processed_data(self):
        df=pd.read_csv(project.ROOT/'data'/'processed'/'player_matches.csv')
        self.assertEqual(df.match_id.nunique(),64)
        self.assertTrue(set(project.POSITION.values()).issubset(set(df.position)))
        self.assertFalse(df.duplicated(['match_id','team','shirt_number']).any())
        self.assertTrue(df.distance_km.between(.1,20).all())
        self.assertTrue(df.minutes_played.gt(0).all())

    def test_prediction_and_bad_json(self):
        sample=json.loads((project.ROOT/'examples'/'player.json').read_text())
        self.assertGreater(project.predict(sample)['distance_km'],0)
        with self.assertRaises(ValueError): project.predict({**sample,'position':'wingback'})
        with self.assertRaises(ValueError): project.predict({**sample,'minutes_played':-1})
        with self.assertRaises(ValueError): project.predict({k:v for k,v in sample.items() if k!='position'})

if __name__=='__main__': unittest.main()
