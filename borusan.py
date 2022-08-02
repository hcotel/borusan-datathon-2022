import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import json
import os
from eda import EDA, TrainTestSplits, LabelEncoders
from metrics import classification_scores, xgb_f1, lgb_f1_score
import eli5
from eli5.sklearn import PermutationImportance
from constants import C
from helper_functions import (
    calculate_optimum_threshold,
    calculate_optimum_threshold_05,
)

def check_fe_ready_file(start_date):
    if os.path.isfile(f"data/train_x_{start_date.strftime('%Y_%m_%d')}.csv"):
        x = pd.read_csv(
            f"data/train_x_{start_date.strftime('%Y_%m_%d')}.csv", index_col="VehicleId"
        )
        y = pd.read_csv(
            f"data/train_y_{start_date.strftime('%Y_%m_%d')}.csv", index_col="VehicleId"
        )
        return x, y, True
    else:
        return None, None, False


def create_train_splits_from_date(start_date):
    train_x, train_y, is_available = check_fe_ready_file(start_date)
    if not is_available or not C.read_ready_files:
        tts = TrainTestSplits(train, start_date)
        train_x, train_y = tts.return_splits()
        eda_train = EDA(train_x, train_y, start_date)
        eda_train.category_encoding(le_dict)
        train_x, train_y = eda_train.return_splits()
        train_x.to_csv(f"data/train_x_{start_date.strftime('%Y_%m_%d')}.csv")
        train_y.to_csv(f"data/train_y_{start_date.strftime('%Y_%m_%d')}.csv")
    return train_x, train_y


def create_train_val_splits_from_date(start_date):
    tts = TrainTestSplits(train, start_date)
    train_x, train_y = tts.return_splits()
    eda_train = EDA(train_x, train_y, start_date)
    eda_train.category_encoding(le_dict)
    train_x, train_y = eda_train.return_splits()

    val_start_date = start_date + datetime.timedelta(days=166)
    tts = TrainTestSplits(train, val_start_date)
    val_x, val_y = tts.return_splits()
    eda_val = EDA(val_x, val_y, val_start_date)
    eda_val.category_encoding(le_dict)
    val_x, val_y = eda_val.return_splits()
    return train_x, train_y, val_x, val_y

sample_submission = pd.read_csv("data/sample_submission.csv").set_index("VehicleId")
train = pd.read_csv("data/train.csv")
train["InvoiceDate"] = pd.to_datetime(train["InvoiceDate"])
train["FirstRegistiration"] = pd.to_datetime(train["FirstRegistiration"])
input_model = pd.read_csv("data/Datathon_Input_Model.csv", encoding="utf-16")
train = train.merge(input_model, on="VehicleId")
if False:
    train = train[:20000]
if C.read_ready_files:
    test_x = pd.read_csv("data/test_x.csv", index_col='VehicleId')
    cat_features = ['City', 'Make', 'BodyCode', 'Serie', 'ModelCode', 'ModelYear', 'BranchCode', 'SplitCode', 'Split_Grubu', 'ItemType', 'MainGroupCode', 'SubGroupCode', 'ServiceInvoiceDate_4', 'dahili_arac', 'ModelDefinition', 'reg_month', 'vehicle_malfunctioned_before', 'customer_malfunctioned_before', 'Milage_adjusted_bin', 'body_type', 'injection_type', 'engine_type', 'boost_type', 'drive_wheels', 'transmission', 'malfunction_VIBRASYON DAMPER', 'malfunction_V KAYIŞI', 'malfunction_GERGİ BİLYASI', 'malfunction_SU POMPASI', 'malfunction_GENLEŞME TANKI', 'malfunction_MOTOR ASKI ROTU', 'malfunction_KRANK SENSÖRÜ', 'malfunction_TURBO', 'malfunction_Egzoz Gazı Turbo Ünitesi Soğutma Sıvısı', 'malfunction_TURBO RADYATÖRÜ', 'is_bakim', 'Make_BodyCode', 'BodyCode_ModelYear', 'BodyCode_ItemType', 'BodyCode_vehicle_malfunctioned_before', 'ModelCode_vehicle_malfunctioned_before', 'ServiceInvoiceDate_4_dahili_arac', 'ModelDefinition_vehicle_malfunctioned_before', 'ModelDefinition_Milage_adjusted_bin']

    features = ['City', 'Make', 'BodyCode', 'Serie', 'ModelCode', 'ModelYear', 'BranchCode', 'Milage', 'SplitCode', 'Split_Grubu', 'ModelDefinition', 'visit_malfunction_ratio', 'ItemType', 'MainGroupCode', 'SubGroupCode', 'ServiceInvoiceDate_4', 'dahili_arac', 'day_since_reg', 'reg_month', 'vehicle_visit_period', 'vehicle_malfunction_period', 'vehicle_malfunction_count', 'vehicle_service_visit_count', 'unique_vehicle_malfunction_count', 'unique_vehicle_service_visit_count', 'vehicle_unique_customer_count', 'customer_unique_vehicle_count', 'unique_customer_malfunction_count', 'unique_customer_service_visit_count', 'customer_visit_malfunction_ratio', 'customer_visit_period', 'customer_malfunction_period', 'customer_malfunction_count', 'customer_service_visit_count', 'vehicle_since_first_inv', 'customer_since_first_inv', 'vehicle_since_last_inv', 'vehicle_day_since_last_malfunction', 'vehicle_day_since_first_malfunction', 'customer_day_since_last_malfunction', 'customer_day_since_first_malfunction', 'vehicle_malfunction_day_last/period', 'customer_malfunction_day_last/period', 'vehicle_malfunctioned_before', 'customer_malfunctioned_before', 'Mile_rate', 'Milage_adjusted', 'Milage_adjusted_bin', 'body_type', 'length_mm', 'width_mm', 'height_mm', 'wheelbase_mm', 'curb_weight_kg', 'payload_kg', 'full_weight_kg', 'maximum_torque_n_m', 'injection_type', 'number_of_cylinders', 'engine_type', 'boost_type', 'engine_hp', 'drive_wheels', 'transmission', 'acceleration_0_100_km/h_s', 'max_speed_km_per_h', 'day_since_VIBRASYON DAMPER', 'malfunction_VIBRASYON DAMPER', 'day_since_V KAYIŞI', 'malfunction_V KAYIŞI', 'day_since_GERGİ BİLYASI', 'malfunction_GERGİ BİLYASI', 'day_since_SU POMPASI', 'malfunction_SU POMPASI', 'day_since_GENLEŞME TANKI', 'malfunction_GENLEŞME TANKI', 'day_since_MOTOR ASKI ROTU', 'malfunction_MOTOR ASKI ROTU', 'day_since_KRANK SENSÖRÜ', 'malfunction_KRANK SENSÖRÜ', 'day_since_TURBO', 'malfunction_TURBO', 'day_since_Egzoz Gazı Turbo Ünitesi Soğutma Sıvısı', 'malfunction_Egzoz Gazı Turbo Ünitesi Soğutma Sıvısı', 'malfunction_TURBO RADYATÖRÜ', 'bakim_count', 'bakim_period', 'day_since_bakim', 'is_bakim', 'Make_BodyCode', 'BodyCode_ModelYear', 'BodyCode_ItemType', 'BodyCode_vehicle_malfunctioned_before', 'ModelCode_vehicle_malfunctioned_before', 'ServiceInvoiceDate_4_dahili_arac', 'ModelDefinition_vehicle_malfunctioned_before', 'ModelDefinition_Milage_adjusted_bin']
else:
    val_start_date = datetime.date(2021, 7, 18)
    tts = TrainTestSplits(train, val_start_date)
    train_x, train_y = tts.return_splits()

    test_start_date = datetime.date(2022, 1, 1)
    tts = TrainTestSplits(train, test_start_date)
    test_x, _ = tts.return_splits()

    eda_test = EDA(test_x, sample_submission, test_start_date)
    test_df = eda_test.df

    cat_features = eda_test.cat_features

    features = eda_test.features
    print(cat_features)

    le = LabelEncoders(test_df, cat_features)
    le_dict = le.encoders_dict

    eda_test.category_encoding(le_dict)
    test_x, _ = eda_test.return_splits()

    test_x.to_csv('data/test_x.csv')
    test_df.to_csv('data/test_df.csv')

train_x, train_y = create_train_splits_from_date(datetime.date(2021, 7, 18))
train_x_all = train_x.copy()
train_y_all = train_y.copy()
print(train_x.shape[0])
for d in [datetime.date(2021, 8, 1), datetime.date(2021, 9, 1), datetime.date(2021, 10, 1), datetime.date(2021, 11, 1), datetime.date(2021, 12, 1), datetime.date(2021, 12, 15)]:
    t_x, t_y = create_train_splits_from_date(d)
    train_x = train_x.append(t_x.loc[t_y[t_y['IsMalfunction'] == 1].index.difference(train_x.index)])
    train_y = train_y.append(t_y.loc[t_y[t_y['IsMalfunction'] == 1].index.difference(train_y.index)])
    print(train_x.shape[0])
train_x.to_csv('data/train_x_plus.csv')
train_y.to_csv('data/train_y_plus.csv')

print("F")

if C.run_fast_trial:
    train_x, train_y = create_train_splits_from_date(datetime.date(2021, 7, 18))
    tr_x, val_x, tr_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=2222, stratify=train_y
    )
    pos_weight = (train_y.count() / train_y.sum())[0] ** 0.5
    if C.model_type == "cat":
        params = {
            "iterations": 1000,
            "random_state": 22,
            "scale_pos_weight": pos_weight,
            "allow_writing_files": False,
            "eval_metric": "F1:use_weights=False",
            "task_type": "GPU",
            "learning_rate": 0.01,
        }
        clf = CatBoostClassifier(**params)

        if C.run_perm_importance:
            clf.fit(tr_x, tr_y, verbose=100)
            perm = PermutationImportance(clf, random_state=22).fit(val_x, val_y)
            print(
                eli5.format_as_text(
                    eli5.explain_weights(
                        perm.estimator, top=1500, feature_names=val_x.columns.tolist()
                    )
                )
            )

        clf.fit(tr_x, tr_y, eval_set=[(val_x, val_y)], verbose=100)
        preds = clf.predict_proba(val_x)[:, 1]
        preds_test = clf.predict_proba(test_x)[:, 1]
        threshold, val_score = calculate_optimum_threshold(val_y, preds)
        preds_test = (preds_test > threshold).astype("int")
        print(f"Test pos rate: {preds_test.sum()}/{preds_test.shape[0]}")

        val_scores = classification_scores(val_y, preds, threshold=threshold)
        print(val_scores)
        test_x["IsMalfunction"] = preds_test
        sample_submission["IsMalfunction"] = test_x["IsMalfunction"]
        sample_submission["IsMalfunction"].to_csv(
            f"submissions/{C.model_type}_{C.trial}_{val_scores['f1']}.csv"
        )

        if C.plot_importance:

            importances = clf.feature_importances_
            indices = np.argsort(importances)
            for indice in indices:
                print(f"'{features[indice]}',")
            indices = indices[:100]

            plt.set_cmap("inferno")
            plt.figure(figsize=(30, 10))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

    with open(
        f"model_parameters/{C.model_type}_{C.trial}_{val_scores['f1']}", "w"
    ) as file:
        file.write(json.dumps(params))

    with open("submissions/submission_notes.csv", "a") as file_object:
        file_object.write(
            f"\nsubmission_{C.model_type}_{C.trial},{val_scores['f1']},{val_scores['accuracy']},{val_scores['roc_auc']},{val_scores['precision']},{val_scores['recall']},{threshold},{params['scale_pos_weight']},{val_scores['pos_rate']},{sample_submission['IsMalfunction'].sum()}"
        )

    exit(1)

if C.validation_type == "stratified":
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=22)
    val_dates = [
        datetime.date(2021, 7, 18),
    ]

    y_test = np.zeros(test_x.shape[0])
    ix = 0
    importance_list = []
    test_prediction_list = []
    val_scores_list = []
    thresholds = []
    y_oofs = []
    train_ys = []
    train_xs = []

    if C.model_type == "cat":
        for val_start_date in val_dates:
            train_x, train_y = create_train_splits_from_date(val_start_date)
            train_xs.append(train_x)
            train_ys.append(train_y)
        train_x = pd.concat(train_xs, axis=0)
        train_y = pd.concat(train_ys, axis=0)
        train_x = train_x[features]
        test_x = test_x[features]
        pos_weight = (train_y.count() / train_y.sum())[0] ** 0.5
        y_oof = np.zeros(train_x.shape[0])
        print(f"Fold: {val_start_date} pos weight:{pos_weight}")
        for train_ind, val_ind in skf.split(train_x, train_y):
            print(f"******* Fold {ix} ******* ")
            tr_x, val_x = (
                train_x.iloc[train_ind].reset_index(drop=True),
                train_x.iloc[val_ind].reset_index(drop=True),
            )
            tr_y, val_y = (
                train_y.iloc[train_ind].reset_index(drop=True),
                train_y.iloc[val_ind].reset_index(drop=True),
            )

            params = {
                "iterations": 3000,
                "random_state": 22,
                #"depth": 5,
                "cat_features": cat_features,
                "scale_pos_weight": 8.1,
                "allow_writing_files": False,
                "learning_rate": 0.01,
                "min_data_in_leaf": 30,
                "l2_leaf_reg": 10,
                #"subsample": 0.9,
                #"colsample_bylevel": 0.8,
                "eval_metric": "F1:use_weights=False",
                "task_type": "GPU",
            }
            clf = CatBoostClassifier(**params)
            clf.fit(
                tr_x,
                tr_y,
                eval_set=[(val_x, val_y)],
                cat_features=cat_features,
                verbose=100,
                early_stopping_rounds=500,
            )

            if C.plot_importance:
                importances = clf.feature_importances_
                importance_list.append(importances)

            preds = clf.predict_proba(val_x)[:, 1]
            threshold, val_score = calculate_optimum_threshold(val_y, preds)
            preds = (preds > threshold).astype("int")
            print(f"Val pos rate: {preds.sum()}/{preds.shape[0]}")
            y_oof[val_ind] = y_oof[val_ind] + preds
            val_scores_list.append(val_score)
            ix = ix + 1

            preds_test = clf.predict_proba(test_x)[:, 1]
            test_prediction_list.append(preds_test)
            preds_test = (preds_test > threshold).astype("int")
            y_test = y_test + preds_test / (N_FOLDS * 1)
            thresholds.append(threshold)

            pass

        #     y_oofs.append(y_oof)
        #     train_ys.append(train_y)
        #
        # y_oof = pd.concat([pd.Series(i) for i in y_oofs], axis=0)
        # train_y = pd.concat([i for i in train_ys], axis=0)
    if C.model_type == "xgb":
        for val_start_date in val_dates:
            train_x, train_y = create_train_splits_from_date(val_start_date)
            train_xs.append(train_x)
            train_ys.append(train_y)
        train_x = pd.concat(train_xs, axis=0)
        train_y = pd.concat(train_ys, axis=0)
        # for f in cat_features:
        #     features.append(f'{f}_count')
        #     features.append(f'{f}_mean')
        #     features.remove(f)
        train_x = train_x[features]
        test_x = test_x[features]
        pos_weight = (train_y.count() / train_y.sum())[0] ** 0.5
        y_oof = np.zeros(train_x.shape[0])
        for train_ind, val_ind in skf.split(train_x, train_y):
            print(f"******* Fold {ix} ******* ")
            tr_x, val_x = (
                train_x.iloc[train_ind].reset_index(drop=True),
                train_x.iloc[val_ind].reset_index(drop=True),
            )
            tr_y, val_y = (
                train_y.iloc[train_ind].reset_index(drop=True),
                train_y.iloc[val_ind].reset_index(drop=True),
            )

            params = {
                "n_estimators": 3000,
                "random_state": 22,
                "cat_features": cat_features,
                "scale_pos_weight": 8,
                "tree_method": "gpu_hist",
                "learning_rate": 0.002,
                "colsample_bytree": 0.6,
                "subsample": 0.9,
                "max_depth": 9,
            }
            xgb_params = params
            xgb_params["eval_metric"] = xgb_f1
            clf = XGBClassifier(**xgb_params)
            clf.fit(
                tr_x,
                tr_y,
                eval_set=[(val_x, val_y)],
                verbose=100,
                early_stopping_rounds=300,
            )

            if C.plot_importance:
                importances = clf.feature_importances_
                importance_list.append(importances)

            preds = clf.predict_proba(val_x)[:, 1]
            threshold, val_score = calculate_optimum_threshold(val_y, preds)
            preds = (preds > threshold).astype("int")
            print(f"Val pos rate: {preds.sum()}/{preds.shape[0]}")
            y_oof[val_ind] = y_oof[val_ind] + preds
            val_scores = classification_scores(train_y, y_oof, threshold=threshold)
            val_scores_list.append(val_score)
            ix = ix + 1

            preds_test = clf.predict_proba(test_x)[:, 1]
            preds_test = (preds_test > threshold).astype("int")
            y_test = y_test + preds_test / (N_FOLDS * 1)
            test_prediction_list.append(preds_test)
            thresholds.append(threshold)

    if C.model_type == "lgb":
        val_start_date = datetime.date(2021, 7, 18)
        train_x = pd.read_csv('data/train_x_plus.csv', index_col='VehicleId')
        train_y = pd.read_csv('data/train_y_plus.csv', index_col='VehicleId')
        test_x.loc[np.isinf(test_x['customer_visit_period']), 'customer_visit_period'] = 10000
        train_x.loc[np.isinf(train_x['customer_visit_period']), 'customer_visit_period'] = 10000
        pos_weight = (train_y.count() / train_y.sum())[0] ** 0.5
        y_oof = np.zeros(train_x.shape[0])
        for train_ind, val_ind in skf.split(train_x, train_y):
            print(f"******* Fold {ix} ******* ")
            tr_x, val_x = (
                train_x.iloc[train_ind].reset_index(drop=True),
                train_x.iloc[val_ind].reset_index(drop=True),
            )
            tr_y, val_y = (
                train_y.iloc[train_ind].reset_index(drop=True),
                train_y.iloc[val_ind].reset_index(drop=True),
            )

            clf = LGBMClassifier(n_estimators=3000, random_state=22, scale_pos_weight=20, learning_rate=0.003, colsample_bytree=0.8, subsample=0.9, max_depth=9,
                                 num_leaves=1000,  reg_lambda=0.05, metric="custom")
            clf.fit(
                tr_x,
                tr_y,
                eval_set=[(val_x, val_y)],
                eval_metric=lgb_f1_score,
                categorical_feature=cat_features,
                verbose=100,
                early_stopping_rounds=300,
            )

            if C.plot_importance:
                importances = clf.feature_importances_
                importance_list.append(importances)

            preds = clf.predict_proba(val_x)[:, 1]
            threshold, val_score = calculate_optimum_threshold(val_y, preds)
            preds = (preds > threshold).astype("int")
            print(f"Val pos rate: {preds.sum()}/{preds.shape[0]}")
            y_oof[val_ind] = y_oof[val_ind] + preds
            val_scores = classification_scores(train_y, y_oof, threshold=threshold)
            val_scores_list.append(val_score)
            ix = ix + 1

            preds_test = clf.predict_proba(test_x)[:, 1]
            preds_test = (preds_test > threshold).astype("int")
            y_test = y_test + preds_test / (N_FOLDS * 1)
            test_prediction_list.append(preds_test)
            thresholds.append(threshold)

    if C.plot_importance:
        importances = np.mean(importance_list, axis=0)
        indices = np.argsort(importances)
        for indice in indices:
            print(f"'{features[indice]}',")

        plt.set_cmap("inferno")
        plt.figure(figsize=(30, 10))
        plt.title("Feature Importances")
        plt.barh(
            range(len(indices)),
            importances[indices],
            color="b",
            align="center",
        )
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.show()

    thresholds = [str(t) for t in thresholds]
    val_scores_list_np = np.array(val_scores_list)
    val_scores_list = [str(t) for t in val_scores_list]
    val_scores = classification_scores(train_y, y_oof, threshold=0.5)
    print(val_scores)
    print(f"Test pos rate: {y_test.sum()}/{y_test.shape[0]}")
    y_test = (y_test >= 0.5).astype("int")
    test_x["IsMalfunction"] = y_test

    # train_x = train_x.append(train_x_all.loc[train_x_all.index.intersection(test_x[test_x['IsMalfunction'] == 1].index.difference(train_x.index))])
    # train_y = train_y.append(train_y_all.loc[train_y_all.index.intersection(test_x[test_x['IsMalfunction'] == 1].index.difference(train_y.index))])
    #
    # train_x.to_csv('data/train_x_pseudo.csv')
    # train_y.to_csv('data/train_y_pseudo.csv')
    # exit(1)

    sample_submission["IsMalfunction"] = test_x["IsMalfunction"]
    sample_submission["IsMalfunction"].to_csv(
        f"submissions/{C.model_type}_cv_{C.trial}_{val_scores['f1']}.csv"
    )

    val_dates = [v.strftime("%Y-%m-%d") for v in val_dates]
    val_dates_str = ";".join(val_dates)
    with open("submissions/submission_notes_cv_2.csv", "a") as file_object:
        file_object.write(
            f"\nsubmission_{C.model_type}_cv_{C.trial},{val_scores['f1']},{val_scores['accuracy']},{val_scores['roc_auc']},{val_scores['precision']},{val_scores['recall']},{'-'.join(thresholds)},{15},{val_scores['pos_rate']},{val_scores_list_np.mean()}+-{val_scores_list_np.std()},{'-'.join(val_scores_list)},{val_dates_str},{sample_submission['IsMalfunction'].sum()}"
        )

elif C.validation_type == "time_splits":
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=2222)
    time_folds = [
        datetime.date(2021, 7, 18),
        datetime.date(2021, 6, 18),
        datetime.date(2021, 5, 18),
    ]

    y_test = np.zeros(test_x.shape[0])
    ix = 0
    importance_list = []
    test_prediction_list = []
    val_scores_list = []
    thresholds = []
    y_oofs = []
    train_ys = []

    ho_x, ho_y = create_train_splits_from_date(datetime.date(2021, 7, 18))
    y_ho = np.zeros(ho_x.shape[0])

    if C.model_type == "cat":
        for time_fold in time_folds:
            # tr_x, tr_y, val_x, val_y = create_train_val_splits_from_date(time_fold)
            tr_x, tr_y = create_train_splits_from_date(time_fold)

            pos_weight = (tr_y.count() / tr_y.sum())[0] ** 0.5
            print(f"Fold: {time_fold} pos weight:{pos_weight}")

            params = {
                "iterations": 1000,
                "random_state": 22,
                "cat_features": cat_features,
                "scale_pos_weight": 8,
                "allow_writing_files": False,
                "learning_rate": 0.01,
                "eval_metric": "F1:use_weights=False",
                "task_type": "GPU",
            }

            clf = CatBoostClassifier(**params)

            if C.run_perm_importance:
                clf.fit(tr_x, tr_y, verbose=100)
                perm = PermutationImportance(clf, random_state=2222).fit(ho_x, ho_y)
                print(
                    eli5.format_as_text(
                        eli5.explain_weights(
                            perm.estimator,
                            top=1500,
                            feature_names=ho_x.columns.tolist(),
                        )
                    )
                )

            clf.fit(tr_x, tr_y, cat_features=cat_features, verbose=100)

            if C.plot_importance:
                importances = clf.feature_importances_
                importance_list.append(importances)

            # preds = clf.predict_proba(val_x)[:, 1]
            # threshold, val_score = calculate_optimum_threshold(val_y, preds)
            # preds = (preds > threshold).astype("int")
            # print(f"Val pos rate: {preds.sum()}/{preds.shape[0]}")
            # val_scores_list.append(val_score)

            threshold = 0.5
            preds_ho = clf.predict_proba(ho_x)[:, 1]
            # preds_ho = (preds_ho >= threshold).astype("int")
            y_ho = y_ho + preds_ho / len(time_folds)

            preds_test = clf.predict_proba(test_x)[:, 1]
            # preds_test = (preds_test >= threshold).astype("int")
            y_test = y_test + preds_test / len(time_folds)
            test_prediction_list.append(preds_test)
            thresholds.append(threshold)

            ix = ix + 1
    if C.model_type == "xgb":
        for train_ind, val_ind in skf.split(train_x, train_y):
            print(f"******* Fold {ix} ******* ")
            tr_x, val_x = (
                train_x.iloc[train_ind].reset_index(drop=True),
                train_x.iloc[val_ind].reset_index(drop=True),
            )
            tr_y, val_y = (
                train_y.iloc[train_ind].reset_index(drop=True),
                train_y.iloc[val_ind].reset_index(drop=True),
            )

            params = {
                "n_estimators": 3000,
                "random_state": 22,
                "cat_features": cat_features,
                "scale_pos_weight": 12,
                "tree_method": "gpu_hist",
                "learning_rate": 0.005,
                "colsample_bytree": 0.6,
                "max_depth": 9,
            }
            xgb_params = params
            xgb_params["eval_metric"] = xgb_f1
            clf = XGBClassifier(**xgb_params)
            clf.fit(
                tr_x,
                tr_y,
                eval_set=[(val_x, val_y)],
                verbose=100,
                early_stopping_rounds=100,
            )

            if C.plot_importance:
                importances = clf.feature_importances_
                importance_list.append(importances)

            preds = clf.predict_proba(val_x)[:, 1]
            threshold, val_score = calculate_optimum_threshold(val_y, preds)
            preds = (preds > threshold).astype("int")
            print(f"Val pos rate: {preds.sum()}/{preds.shape[0]}")
            val_scores_list.append(val_score)
            ix = ix + 1

            preds_test = clf.predict_proba(test_x)[:, 1]
            preds_test = (preds_test > threshold).astype("int")
            y_test = y_test + preds_test / len(time_folds)
            test_prediction_list.append(preds_test)
            thresholds.append(threshold)

    if C.plot_importance:
        importances = np.mean(importance_list, axis=0)
        indices = np.argsort(importances)
        for indice in indices:
            print(f"'{features[indice]}',")

        plt.set_cmap("inferno")
        plt.figure(figsize=(30, 10))
        plt.title("Feature Importances")
        plt.barh(
            range(len(indices)),
            importances[indices],
            color="b",
            align="center",
        )
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.show()

    thresholds = [str(t) for t in thresholds]
    # val_scores_list = np.array(val_scores_list)
    threshold, val_score = calculate_optimum_threshold(ho_y, y_ho)
    val_scores = classification_scores(ho_y, y_ho, threshold=threshold)
    print(val_scores)
    print(f"Test pos rate: {y_test.sum()}/{y_test.shape[0]}")
    y_test = (y_test >= 0.5).astype("int")
    test_x["IsMalfunction"] = y_test
    sample_submission["IsMalfunction"] = test_x["IsMalfunction"]
    sample_submission["IsMalfunction"].to_csv(
        f"submissions/{C.model_type}_tf_{C.trial}_{val_scores['f1']}.csv"
    )

    with open("submissions/submission_notes_tf.csv", "a") as file_object:
        file_object.write(
            f"\nsubmission_{C.model_type}_tf_{C.trial},{val_scores['f1']},{val_scores['accuracy']},{val_scores['roc_auc']},{val_scores['precision']},{val_scores['recall']},{'-'.join(thresholds)},{params['scale_pos_weight']},{val_scores['pos_rate']},,{sample_submission['IsMalfunction'].sum()}"
        )

    with open(
        f"model_parameters/{C.model_type}_tf_{C.trial}_{val_scores['f1']}", "w"
    ) as file:
        file.write(json.dumps(params))
