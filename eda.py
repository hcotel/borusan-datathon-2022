import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import datetime
from itertools import combinations, product
import bisect
import warnings
from helper_functions import overwrite_car_metrics

warnings.filterwarnings("ignore")

sample_submission = pd.read_csv("data/sample_submission.csv").set_index("VehicleId")
car_db_metric = pd.read_csv("data/car_db_metric.csv")
car_db_metric = overwrite_car_metrics(car_db_metric)
train_y = sample_submission.copy()


class LabelEncoders:
    def __init__(self, df, cat_features):
        self.df = df
        self.cat_features = cat_features
        self.encoders_dict = self.fit_encoders()

    def fit_encoders(self):
        encoders_dict = {}
        for c in self.cat_features:
            le = LabelEncoder()
            le.fit(self.df[c])
            if c == "ModelCode_vehicle_malfunctioned_before":
                usual_suspects = ["VN71_0", "JA81_0", "XV31_0", "WG41_0"]
                le_classes = le.classes_.tolist()
                for s in usual_suspects:
                    if s not in le_classes:
                        bisect.insort_left(le_classes, s)

                le.classes_ = le_classes
            if c == "ModelDefinition_Milage_adjusted_bin":
                le_classes = []
                le_classes_1 = encoders_dict["ModelDefinition"].classes_.tolist()
                le_classes_2 = encoders_dict["Milage_adjusted_bin"].classes_.tolist()
                le_classes_12 = list(product(le_classes_1, le_classes_2))
                for cl, b in le_classes_12:
                    bisect.insort_left(le_classes, f"{cl}_{b}")
                le.classes_ = le_classes
            encoders_dict[c] = le
        return encoders_dict


class EDA:
    def __init__(self, df, y, origin_date):
        self.df = df
        self.y = y
        self.origin_date = origin_date
        self.overwrite_values()
        self.merge_car_metrics()
        self.features = self.form_features()
        self.cat_features = self.form_cat_features()
        self.df = self.create_vehicle_features()
        self.df = self.create_customer_features()
        self.create_new_features()
        self.create_older_malfunction_analysis()
        self.create_bakim_related_features()
        self.create_milage_features_v2()
        self.df = self.create_category_interaction_features(self.df)
        self.x = self.group_by_vehicle()

    def form_features(self):
        features = [
            "City",
            "Make",
            "BodyCode",
            "Serie",
            "ModelCode",
            "ModelYear",  #'ServiceInvoiceDate',
            "BranchCode",
            "Milage",
            "SplitCode",
            "Split_Grubu",  #'ServiceInvoiceLineNo', 'ItemNo',
            "ModelDefinition",
            "visit_malfunction_ratio",
            "ItemType",
            "MainGroupCode",
            "SubGroupCode",
            "ServiceInvoiceDate_4",
            "dahili_arac",
            "day_since_reg",
            "reg_month",
            "vehicle_visit_period",
            "vehicle_malfunction_period",
            "vehicle_malfunction_count",
            "vehicle_service_visit_count",
            "unique_vehicle_malfunction_count",
            "unique_vehicle_service_visit_count",
            "vehicle_unique_customer_count",
            "customer_unique_vehicle_count",
            "unique_customer_malfunction_count",
            "unique_customer_service_visit_count",
            "customer_visit_malfunction_ratio",
            "customer_visit_period",
            "customer_malfunction_period",
            "customer_malfunction_count",
            "customer_service_visit_count",
            "vehicle_since_first_inv",
            "customer_since_first_inv",
            "vehicle_since_last_inv",
            "vehicle_day_since_last_malfunction",
            "vehicle_day_since_first_malfunction",
            "customer_day_since_last_malfunction",
            "customer_day_since_first_malfunction",
            "vehicle_malfunction_day_last/period",
            "customer_malfunction_day_last/period",
            "vehicle_malfunctioned_before",
            "customer_malfunctioned_before",
            "Mile_rate",
            "Milage_adjusted",
            "Milage_adjusted_bin",
            "body_type",
            "length_mm",
            "width_mm",
            "height_mm",
            "wheelbase_mm",
            "curb_weight_kg",
            "payload_kg",
            "full_weight_kg",
            "maximum_torque_n_m",
            "injection_type",
            "number_of_cylinders",
            "engine_type",
            "boost_type",
            "engine_hp",
            "drive_wheels",
            "transmission",
            "acceleration_0_100_km/h_s",
            "max_speed_km_per_h",
        ]
        return features

    def form_cat_features(self):
        cat_features = [
            "City",
            "Make",
            "BodyCode",
            "Serie",
            "ModelCode",
            "ModelYear",  #'ServiceInvoiceDate',
            "BranchCode",
            "SplitCode",
            "Split_Grubu",  # "ServiceInvoiceLineNo",
            # "ItemNo",
            "ItemType",
            "MainGroupCode",
            "SubGroupCode",
            "ServiceInvoiceDate_4",
            "dahili_arac",
            "ModelDefinition",
            "reg_month",
            "vehicle_malfunctioned_before",
            "customer_malfunctioned_before",
            "Milage_adjusted_bin",
            "body_type",
            "injection_type",
            "engine_type",
            "boost_type",
            "drive_wheels",
            "transmission",
        ]
        cat_features = [c for c in cat_features if c in self.features]
        return cat_features

    def merge_car_metrics(self):
        self.df = self.df.merge(
            car_db_metric[
                [
                    "make",
                    "model",
                    "modeldef",
                    "body_type",
                    "length_mm",
                    "width_mm",
                    "height_mm",
                    "wheelbase_mm",
                    "curb_weight_kg",
                    "payload_kg",
                    "full_weight_kg",
                    "maximum_torque_n_m",
                    "injection_type",
                    "number_of_cylinders",
                    "engine_type",
                    "boost_type",
                    "engine_hp",
                    "drive_wheels",
                    "transmission",
                    "acceleration_0_100_km/h_s",
                    "max_speed_km_per_h",
                ]
            ],
            left_on=["Make", "Serie", "ModelDefinition"],
            right_on=["make", "model", "modeldef"],
            how="left",
        )
        pass

    def category_encoding(self, le_dict):
        for categorical_feature in self.cat_features:
            print(categorical_feature)
            self.df[categorical_feature] = (
                le_dict[categorical_feature]
                .transform(self.df[categorical_feature])
                .astype("int16")
            )
        self.x = self.group_by_vehicle()
        for categorical_feature in self.cat_features:
            cat_stat = (
                self.x.groupby(categorical_feature)["vehicle_malfunctioned_before"]
                .agg(["count", "mean"])
                .rename(
                    columns={
                        "count": f"{categorical_feature}_count",
                        "mean": f"{categorical_feature}_mean",
                    }
                )
            )
            self.x = self.x.merge(
                cat_stat, left_on=categorical_feature, right_index=True, how="left"
            )
            self.features.append(f"{categorical_feature}_count")
            self.features.append(f"{categorical_feature}_mean")
        pass

    def overwrite_values(self):
        self.df.loc[self.df["SplitCode"] == "other", "SplitCode"] = 555
        self.df["SplitCode"] = self.df["SplitCode"].astype("int16")
        self.df["ServiceInvoiceLineNo"] = self.df["ServiceInvoiceLineNo"].astype("str")
        self.df = self.df.replace("Belirsiz", np.nan)
        self.df.loc[self.df["Make"] == "Land Rover", "Make"] = "LAND ROVER"
        self.df.loc[self.df["BodyCode"] == "g11", "BodyCode"] = "G11"
        self.df.loc[self.df["BodyCode"] == "g12", "BodyCode"] = "G12"
        self.df.loc[self.df["BodyCode"] == "l494", "BodyCode"] = "L494"
        self.df.loc[self.df["Milage"] == 536666, "Milage"] = 53666
        self.df.loc[
            self.df["FirstRegistiration"].isna(), "FirstRegistiration"
        ] = pd.to_datetime(self.df["ModelYear"]) + pd.Timedelta(182, unit="d")

        self.df.loc[
            self.df["ModelDefinition"].isna(), "ModelDefinition"
        ] = "BMW X1 sDrive18i"
        self.df["ModelDefinition"] = self.df["ModelDefinition"].str.lstrip("Yeni ")
        self.df["ModelDefinition"] = (
            self.df["ModelDefinition"].str.rstrip("..").str.rstrip(".")
        )
        self.df["ModelDefinition"] = self.df["ModelDefinition"].str.replace("é", "e")
        self.df.loc[
            self.df["ModelDefinition"] == "BMW116i", "ModelDefinition"
        ] = "BMW 116i"
        self.df.loc[
            self.df["ModelDefinition"] == "BMW 520i Sedan KOSİFLER ANTALYA TEST",
            "ModelDefinition",
        ] = "BMW 520i Sedan"
        self.df.loc[
            self.df["ModelDefinition"] == "BMW 218i Gran Coupé", "ModelDefinition"
        ] = "BMW 218i Gran Coupe"
        self.df["ModelDefinition"] = self.df["ModelDefinition"].str.lower()
        self.df["ModelDefinition"] = self.df["ModelDefinition"].str.replace("ı", "i")
        self.df["ModelDefinition"] = self.df["ModelDefinition"].str.replace(
            "x drive", "xdrive"
        )
        self.df["ModelDefinition"] = self.df["ModelDefinition"].str.replace(
            " sedan", ""
        )
        self.df["ModelDefinition"] = self.df["ModelDefinition"].str.replace(" hp", "hp")
        self.df.loc[
            self.df["ModelDefinition"] == "range rover velar 2.0d td4 180hp aw",
            "ModelDefinition",
        ] = "range rover velar 2.0d td4 180hp awd"
        self.df.loc[
            self.df["ModelDefinition"] == "range rover sport 2.0 sd4", "ModelDefinition"
        ] = "range rover sport 2.0 sd4 240hp awd"

        self.df.loc[
            self.df["ItemDescription"] == "GERGI BILYAS", "ItemDescription"
        ] = "GERGİ BİLYASI"
        self.df.loc[
            self.df["ItemDescription"] == "V KAYIŞ", "ItemDescription"
        ] = "V KAYIŞI"
        self.df.loc[
            self.df["ItemDescription"] == "GERGİ", "ItemDescription"
        ] = "GERGİ BİLYASI"
        self.df.loc[
            self.df["ItemDescription"] == "V KAYISI", "ItemDescription"
        ] = "V KAYIŞI"
        self.df.loc[
            self.df["ItemDescription"] == "GERGI", "ItemDescription"
        ] = "GERGİ BİLYASI"
        self.df.loc[
            self.df["ItemDescription"] == "TİTREŞİM DAMPERİ", "ItemDescription"
        ] = "VIBRASYON DAMPER"
        self.df.loc[
            self.df["ItemDescription"] == "Dönel Titreşim Damperinin Değiştirilmesi",
            "ItemDescription",
        ] = "VIBRASYON DAMPER"
        self.df.loc[
            self.df["ItemDescription"] == "Dönel Titreşim Damperini Değiştirme",
            "ItemDescription",
        ] = "VIBRASYON DAMPER"
        pass

    def create_category_interaction_features(self, df, add_cat_features=True):
        cat_features_level_2 = list(combinations(self.cat_features, 2))
        category_interaction_features = [
            "ModelYear_BodyCode",
            "ModelCode_vehicle_malfunctioned_before",
            "BodyCode_vehicle_malfunctioned_before",
            "ModelDefinition_vehicle_malfunctioned_before",
            "BodyCode_ModelYear",
            "ModelDefinition_Milage_adjusted_bin",
            "Make_BodyCode",
            "BodyCode_ItemType",
            "ServiceInvoiceDate_4_dahili_arac",
        ]
        for pair in cat_features_level_2:
            if f"{pair[0]}_{pair[1]}" in category_interaction_features:
                df[f"{pair[0]}_{pair[1]}"] = (
                    df[pair[0]].astype("str") + "_" + df[pair[1]].astype("str")
                )
                if add_cat_features:
                    self.features.append(f"{pair[0]}_{pair[1]}")
                    self.cat_features.append(f"{pair[0]}_{pair[1]}")
        return df

    def create_new_features(self):
        self.df["ServiceInvoiceDate_4"] = self.df["ServiceInvoiceDate"].str[:4]
        self.df["dahili_arac"] = (self.df["CustomerId"] == "DAHİLİ").astype("int8")
        self.df["day_since_reg"] = (
            self.origin_date - self.df["FirstRegistiration"].dt.date
        ).dt.days
        self.df.loc[self.df["day_since_reg"] < 0, "day_since_reg"] = 0
        self.df["vehicle_since_first_inv"] = (
            self.origin_date - self.df["vehicle_first_invoice_date"].dt.date
        ).dt.days
        self.df["customer_since_first_inv"] = (
            self.origin_date - self.df["customer_first_invoice_date"].dt.date
        ).dt.days
        self.df["reg_month"] = self.df["FirstRegistiration"].dt.month

    def create_older_malfunction_analysis(self):
        malfunction_ban_features = [
            "day_since_TURBO RADYATÖRÜ",
            "malfunction_SU POMPA KAYIŞI",
            "day_since_SU POMPA KAYIŞI",
        ]
        malfunction_df = self.df[self.df["IsMalfunction"] == 1]
        malfunction_item_description_list = [
            "VIBRASYON DAMPER",
            "V KAYIŞI",
            "GERGİ BİLYASI",
            "SU POMPASI",
            "GENLEŞME TANKI",
            "MOTOR ASKI ROTU",
            "KRANK SENSÖRÜ",
            "TURBO",
            "Egzoz Gazı Turbo Ünitesi Soğutma Sıvısı",
            "TURBO RADYATÖRÜ",
            "SU POMPA KAYIŞI",
        ]
        for item_desc in malfunction_item_description_list:
            malfunction_item_last_by_vehicle = pd.DataFrame(
                malfunction_df[malfunction_df["ItemDescription"] == item_desc]
                .groupby("VehicleId")["InvoiceDate"]
                .max()
            ).rename(columns={"InvoiceDate": f"{item_desc}_last"})
            if malfunction_item_last_by_vehicle.shape[0] > 0:
                malfunction_item_last_by_vehicle[f"day_since_{item_desc}"] = (
                    self.origin_date
                    - malfunction_item_last_by_vehicle[f"{item_desc}_last"].dt.date
                ).dt.days
                malfunction_item_last_by_vehicle[f"malfunction_{item_desc}"] = (
                    malfunction_item_last_by_vehicle[f"day_since_{item_desc}"].notna()
                ).astype("int")
                self.df = self.df.merge(
                    malfunction_item_last_by_vehicle, on="VehicleId", how="left"
                )
            else:
                self.df[f"day_since_{item_desc}"] = 10000
                self.df[f"malfunction_{item_desc}"] = 0
            self.df[f"day_since_{item_desc}"] = self.df[
                f"day_since_{item_desc}"
            ].fillna(10000)
            if f"day_since_{item_desc}" not in malfunction_ban_features:
                self.features.append(f"day_since_{item_desc}")
            if f"malfunction_{item_desc}" not in malfunction_ban_features:
                self.features.append(f"malfunction_{item_desc}")
                self.cat_features.append(f"malfunction_{item_desc}")

    def create_bakim_related_features(self):
        bakim_df = self.df[self.df["MainGroupDescription"] == "Bakım"]
        bakim_by_vehicle = (
            bakim_df.groupby("VehicleId")["InvoiceDate"]
            .agg(["min", "max"])
            .rename(columns={"min": f"bakim_first", "max": f"bakim_last"})
        )
        bakim_by_vehicle_count = pd.DataFrame(
            bakim_df.groupby("VehicleId")["InvoiceDate"].nunique()
        ).rename(columns={"InvoiceDate": f"bakim_count"})
        bakim_by_vehicle = bakim_by_vehicle.merge(
            bakim_by_vehicle_count, on="VehicleId", how="left"
        )
        bakim_by_vehicle["bakim_period"] = (
            bakim_by_vehicle["bakim_last"] - bakim_by_vehicle["bakim_first"]
        ).dt.days / bakim_by_vehicle["bakim_count"]
        bakim_by_vehicle["day_since_bakim"] = (
            self.origin_date - bakim_by_vehicle["bakim_last"].dt.date
        ).dt.days
        bakim_by_vehicle["is_bakim"] = (
            bakim_by_vehicle["day_since_bakim"].notna()
        ).astype("int")
        bakim_by_vehicle.loc[
            bakim_by_vehicle["bakim_count"] == 0, "bakim_period"
        ] = 10000
        self.df = self.df.merge(bakim_by_vehicle, on="VehicleId", how="left")
        self.df.loc[self.df["bakim_count"] == 0, "day_since_bakim"] = self.df[
            "day_since_reg"
        ]
        self.features.append("bakim_count")
        self.features.append("bakim_period")
        self.features.append("day_since_bakim")
        self.features.append("is_bakim")
        self.cat_features.append("is_bakim")

    def create_milage_features(self):
        self.df.loc[self.df["Milage"] == 1000000, "Milage"] = np.nan
        self.df.loc[self.df["Milage"] < 100, "Milage"] = 0
        inv_stat = (
            self.df.groupby("VehicleId")["InvoiceDate"]
            .agg(["min", "max"])
            .rename(columns={"min": f"inv_first", "max": f"inv_last"})
        )
        mile_stat = (
            self.df.groupby("VehicleId")["Milage"]
            .agg(["min", "max"])
            .rename(columns={"min": f"mile_first", "max": f"mile_last"})
        )
        vehicle_reg = self.df[["VehicleId", "FirstRegistiration"]].drop_duplicates()
        mile_stat = mile_stat.merge(inv_stat, on="VehicleId", how="left").merge(
            vehicle_reg, on="VehicleId", how="left"
        )
        mile_stat["Mile_rate"] = (mile_stat["mile_last"] - mile_stat["mile_first"]) / (
            mile_stat["inv_last"] - mile_stat["inv_first"]
        ).dt.days
        mile_stat.loc[mile_stat["Mile_rate"].isna(), "Mile_rate"] = (
            mile_stat["mile_last"]
            / (mile_stat["inv_last"] - mile_stat["FirstRegistiration"]).dt.days
        )
        mile_stat.loc[mile_stat["Mile_rate"] == np.inf, "Mile_rate"] = (
            mile_stat["mile_last"]
            / (mile_stat["inv_last"] - mile_stat["FirstRegistiration"]).dt.days
        )
        mile_stat.loc[mile_stat["Mile_rate"].isna(), "Mile_rate"] = 0
        mile_stat.loc[mile_stat["Mile_rate"] == np.inf, "Mile_rate"] = 0
        mile_stat.loc[mile_stat["Mile_rate"] < 0, "Mile_rate"] = 0
        mile_stat.loc[mile_stat["Mile_rate"] > 200, "Mile_rate"] = 200
        mile_stat["Milage_adjusted"] = (
            (self.origin_date - mile_stat["inv_last"].dt.date).dt.days + 166
        ) * mile_stat["Mile_rate"] + mile_stat["mile_last"]
        mile_stat["Milage_adjusted_bin"] = pd.cut(
            mile_stat["Milage_adjusted"],
            bins=[-10, 25000, 50000, 75000, 100000, 100000000],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype("int")
        self.df = self.df.merge(
            mile_stat[
                ["VehicleId", "Milage_adjusted", "Milage_adjusted_bin", "Mile_rate"]
            ],
            on="VehicleId",
            how="left",
        )
        pass

    def create_milage_features_v2(self):
        self.df.loc[self.df["Milage"] == 1000000, "Milage"] = np.nan
        self.df.loc[self.df["Milage"] < 100, "Milage"] = 0
        milage_df = self.df.iloc[
            self.df[["VehicleId", "InvoiceDate"]].drop_duplicates().index
        ][["VehicleId", "Milage", "InvoiceDate", "FirstRegistiration"]]
        milage_df = (
            milage_df.sort_values("InvoiceDate", ascending=False)
            .groupby("VehicleId")
            .head(2)
        )
        milage_df = milage_df.sort_values(["VehicleId", "InvoiceDate"], ascending=False)
        milage_df["Milage-1"] = milage_df["Milage"].shift(-1)
        milage_df["InvoiceDate-1"] = milage_df["InvoiceDate"].shift(-1)
        milage_df_count = pd.DataFrame(
            milage_df.groupby("VehicleId")["InvoiceDate"].count()
        ).rename(columns={"InvoiceDate": f"count"})
        milage_df_count = milage_df_count[milage_df_count["count"] > 1]
        multiple_vehicle_list = (
            milage_df_count.reset_index()["VehicleId"].unique().tolist()
        )
        milage_df = (
            milage_df[milage_df["VehicleId"].isin(multiple_vehicle_list)]
            .sort_values("InvoiceDate", ascending=False)
            .groupby("VehicleId")
            .head(1)
        )

        milage_df["mile_diff"] = milage_df["Milage"] - milage_df["Milage-1"]
        milage_df["inv_diff"] = (
            milage_df["InvoiceDate"] - milage_df["InvoiceDate-1"]
        ).dt.days
        milage_df.loc[milage_df["inv_diff"] > 20, "Mile_rate"] = (
            milage_df["mile_diff"] / milage_df["inv_diff"]
        )
        milage_df.loc[milage_df["Mile_rate"] > 200, "Mile_rate"] = np.nan

        milage_max = (
            milage_df.groupby("VehicleId")[["Milage", "InvoiceDate"]]
            .max()
            .rename(columns={"InvoiceDate": f"inv_last", "Milage": f"milage_last"})
        )
        milage_df = milage_df.merge(milage_max, on="VehicleId", how="left")
        milage_df.loc[milage_df["Mile_rate"].isna(), "Mile_rate"] = (
            milage_df["milage_last"]
            / (milage_df["inv_last"] - milage_df["FirstRegistiration"]).dt.days
        )
        milage_df.loc[milage_df["Mile_rate"] <= 0, "Mile_rate"] = 0
        milage_df.loc[milage_df["Mile_rate"].isna(), "Mile_rate"] = 0

        milage_df["Milage_adjusted"] = (
            (self.origin_date - milage_df["inv_last"].dt.date).dt.days + 166
        ) * milage_df["Mile_rate"] + milage_df["milage_last"]
        milage_df.loc[milage_df["Milage_adjusted"].isna(), "Milage_adjusted"] = 0
        milage_df.loc[milage_df["Milage"].isna(), "Milage"] = 0
        milage_df["Milage_adjusted_bin"] = pd.cut(
            milage_df["Milage_adjusted"],
            bins=[-10, 25000, 50000, 75000, 100000, 100000000],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype("int")

        self.df = self.df.merge(
            milage_df[
                ["VehicleId", "Milage_adjusted", "Milage_adjusted_bin", "Mile_rate"]
            ],
            on="VehicleId",
            how="left",
        )
        pass

    def create_vehicle_features(self):
        vehicle_stats = (
            self.df.groupby("VehicleId")["IsMalfunction"]
            .agg(["sum", "count"])
            .rename(
                columns={
                    "sum": "vehicle_malfunction_count",
                    "count": "vehicle_service_visit_count",
                }
            )
        )
        vehicle_stats_unique = (
            self.df[["VehicleId", "InvoiceDate", "IsMalfunction"]]
            .drop_duplicates()
            .groupby("VehicleId")["IsMalfunction"]
            .agg(["sum", "count"])
            .rename(
                columns={
                    "sum": "unique_vehicle_malfunction_count",
                    "count": "unique_vehicle_service_visit_count",
                }
            )
        )
        vehicle_stats_unique["visit_malfunction_ratio"] = (
            vehicle_stats_unique["unique_vehicle_malfunction_count"]
            / vehicle_stats_unique["unique_vehicle_service_visit_count"]
        )
        vehicle_unique_customers = pd.DataFrame(
            self.df.groupby("VehicleId")["CustomerId"].nunique()
        ).rename(columns={"CustomerId": "vehicle_unique_customer_count"})
        vehicle_stats["vehicle_malfunctioned_before"] = (
            vehicle_stats["vehicle_malfunction_count"] > 0
        ).astype("int")
        vehicle_first = (
            self.df.groupby("VehicleId")["InvoiceDate"]
            .agg(["min", "max"])
            .rename(
                columns={
                    "min": "vehicle_first_invoice_date",
                    "max": "vehicle_last_invoice_date",
                }
            )
        )
        vehicle_first["vehicle_since_last_inv"] = (
            self.origin_date - vehicle_first["vehicle_last_invoice_date"].dt.date
        ).dt.days
        vehicle_first["vehicle_last_first_diff"] = (
            vehicle_first["vehicle_last_invoice_date"]
            - vehicle_first["vehicle_first_invoice_date"]
        ).dt.days
        vehicle_malfunction_stats = (
            self.df[self.df["IsMalfunction"] == 1]
            .groupby("VehicleId")["InvoiceDate"]
            .agg(["min", "max"])
            .rename(
                columns={
                    "min": "vehicle_first_malfunction_date",
                    "max": "vehicle_last_malfunction_date",
                }
            )
        )
        vehicle_malfunction_stats["vehicle_day_since_last_malfunction"] = (
            self.origin_date
            - vehicle_malfunction_stats["vehicle_last_malfunction_date"].dt.date
        ).dt.days
        vehicle_malfunction_stats["vehicle_day_since_first_malfunction"] = (
            self.origin_date
            - vehicle_malfunction_stats["vehicle_first_malfunction_date"].dt.date
        ).dt.days
        df = (
            self.df.merge(
                vehicle_stats, left_on="VehicleId", right_index=True, how="left"
            )
            .merge(vehicle_first, left_on="VehicleId", right_index=True, how="left")
            .merge(
                vehicle_malfunction_stats,
                left_on="VehicleId",
                right_index=True,
                how="left",
            )
            .merge(
                vehicle_stats_unique, left_on="VehicleId", right_index=True, how="left"
            )
            .merge(
                vehicle_unique_customers,
                left_on="VehicleId",
                right_index=True,
                how="left",
            )
        )
        df["vehicle_visit_period"] = (
            df["vehicle_last_first_diff"] / df["unique_vehicle_service_visit_count"]
        )
        df["vehicle_malfunction_period"] = (
            df["vehicle_last_first_diff"] / df["unique_vehicle_malfunction_count"]
        )
        df["vehicle_malfunction_day_last/period"] = (
            df["vehicle_day_since_last_malfunction"] / df["vehicle_malfunction_period"]
        )
        df.loc[df["vehicle_service_visit_count"] == 0, "vehicle_visit_period"] = 10000
        df.loc[
            df["vehicle_malfunction_count"] == 0, "vehicle_malfunction_period"
        ] = 10000
        df.loc[
            df["vehicle_malfunction_period"] == 0, "vehicle_malfunction_day_last/period"
        ] = 0
        return df

    def create_customer_features(self):
        customer_stats = (
            self.df[~(self.df["CustomerId"] == "DAHİLİ")]
            .groupby("CustomerId")["IsMalfunction"]
            .agg(["sum", "count"])
            .rename(
                columns={
                    "sum": "customer_malfunction_count",
                    "count": "customer_service_visit_count",
                }
            )
        )
        customer_stats["customer_malfunctioned_before"] = (
            customer_stats["customer_malfunction_count"] > 0
        ).astype("int")
        customer_stats_unique = (
            self.df[["CustomerId", "InvoiceDate", "IsMalfunction"]]
            .drop_duplicates()
            .groupby("CustomerId")["IsMalfunction"]
            .agg(["sum", "count"])
            .rename(
                columns={
                    "sum": "unique_customer_malfunction_count",
                    "count": "unique_customer_service_visit_count",
                }
            )
        )
        customer_stats_unique["customer_visit_malfunction_ratio"] = (
            customer_stats_unique["unique_customer_malfunction_count"]
            / customer_stats_unique["unique_customer_service_visit_count"]
        )
        customer_unique_vehicles = pd.DataFrame(
            self.df.groupby("CustomerId")["VehicleId"].nunique()
        ).rename(columns={"VehicleId": "customer_unique_vehicle_count"})
        customer_first = (
            self.df[~(self.df["CustomerId"] == "DAHİLİ")]
            .groupby("CustomerId")["InvoiceDate"]
            .agg(["min", "max"])
            .rename(
                columns={
                    "min": "customer_first_invoice_date",
                    "max": "customer_last_invoice_date",
                }
            )
        )
        customer_first["customer_last_first_diff"] = (
            customer_first["customer_last_invoice_date"]
            - customer_first["customer_first_invoice_date"]
        ).dt.days
        customer_malfunction_stats = (
            self.df[
                (self.df["IsMalfunction"] == 1) & (~(self.df["CustomerId"] == "DAHİLİ"))
            ]
            .groupby("CustomerId")["InvoiceDate"]
            .agg(["min", "max"])
            .rename(
                columns={
                    "min": "customer_first_malfunction_date",
                    "max": "customer_last_malfunction_date",
                }
            )
        )
        customer_malfunction_stats["customer_day_since_last_malfunction"] = (
            self.origin_date
            - customer_malfunction_stats["customer_last_malfunction_date"].dt.date
        ).dt.days
        customer_malfunction_stats["customer_day_since_first_malfunction"] = (
            self.origin_date
            - customer_malfunction_stats["customer_first_malfunction_date"].dt.date
        ).dt.days
        df = (
            self.df.merge(
                customer_stats, left_on="CustomerId", right_index=True, how="left"
            )
            .merge(customer_first, left_on="CustomerId", right_index=True, how="left")
            .merge(
                customer_malfunction_stats,
                left_on="CustomerId",
                right_index=True,
                how="left",
            )
            .merge(
                customer_stats_unique,
                left_on="CustomerId",
                right_index=True,
                how="left",
            )
            .merge(
                customer_unique_vehicles,
                left_on="CustomerId",
                right_index=True,
                how="left",
            )
        )
        df["customer_visit_period"] = (
            df["customer_last_first_diff"] / df["unique_customer_service_visit_count"]
        )
        df["customer_malfunction_period"] = (
            df["customer_last_first_diff"] / df["unique_customer_malfunction_count"]
        )
        df["customer_malfunction_day_last/period"] = (
            df["customer_day_since_last_malfunction"]
            / df["customer_malfunction_period"]
        )
        df.loc[df["customer_service_visit_count"] == 0, "customer_visit_period"] = 10000
        df.loc[
            df["customer_malfunction_count"] == 0, "customer_malfunction_period"
        ] = 10000
        df.loc[
            df["customer_malfunction_period"] == 0,
            "customer_malfunction_day_last/period",
        ] = 0
        return df

    def group_by_vehicle(self):
        x = self.df.groupby("VehicleId").tail(1).set_index("VehicleId")
        x = self.create_post_groupby_features(x)
        # x = x[self.features]
        return x

    def create_post_groupby_features(self, x):
        # x = self.create_category_interaction_features(x)
        # x = self.sparse_category_encoding(x)
        return x

    def return_splits(self):
        return self.x.sort_index(), self.y.sort_index()


class TrainTestSplits:
    def __init__(self, df, test_start_date):
        self.df = df
        self.test_start_date = test_start_date
        self.train_y_base = self.calculate_y_base()
        self.train_df, self.vehicle_list = self.calculate_vehicle_list()
        self.train_y = self.calculate_y()

    def calculate_y_base(self):
        sample_submission = (
            pd.read_csv("data/sample_submission.csv").set_index("VehicleId").fillna(0)
        )
        return sample_submission

    def calculate_vehicle_list(self):
        df_train = self.df[self.df["InvoiceDate"].dt.date < self.test_start_date]
        vehicle_list = df_train["VehicleId"].unique()
        return df_train, vehicle_list

    def calculate_y(self):
        test_end_date = self.test_start_date + timedelta(days=166)
        test_range = self.df[
            (self.df["InvoiceDate"].dt.date < test_end_date)
            & (self.df["InvoiceDate"].dt.date >= self.test_start_date)
        ]
        test_range = pd.DataFrame(
            test_range.groupby("VehicleId")["IsMalfunction"].mean()
        )
        test_range["IsMalfunction"] = (test_range["IsMalfunction"] > 0).astype("int8")
        train_y = self.train_y_base[self.train_y_base.index.isin(self.vehicle_list)]
        train_y["IsMalfunction"] = test_range["IsMalfunction"]
        train_y = train_y.fillna(0)
        return train_y

    def return_splits(self):
        return self.train_df, self.train_y
