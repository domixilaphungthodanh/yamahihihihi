# Kết quả đánh giá

Mô hình chọn bằng cross-validation: **random_forest**.

Train: 318 xe; test: 80 xe.

MAE test: **0.833 L/100 km**. RMSE: 1.201; R²: 0.906.

MAE baseline test: 3.145 L/100 km.

        model   cv_mae   cv_std                                               parameters
random_forest 0.926575 0.107210 {"model__max_depth": null, "model__min_samples_leaf": 1}
        ridge 0.990205 0.122347                                     {"model__alpha": 10}
     baseline 3.231143 0.186889                                                       {}

Permutation importance được tính trên train, chỉ mang tính khám phá và có thể lạc quan; không chứng minh quan hệ nhân quả.

Dữ liệu xe 1970–1982 trong chu trình đô thị; chưa được xác thực cho xe hiện đại, xe điện hoặc chuyến đi thực tế. Chia ngẫu nhiên đánh giá xe trong cùng phân phối lịch sử, không đánh giá khả năng dự đoán tương lai.
