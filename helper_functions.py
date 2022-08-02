from sklearn.metrics import f1_score
import numpy as np

def calculate_optimum_threshold(y_true, y_pred_labels):
    max_thresh = 0.3
    max_score = 0
    for thresh in range(30, 60, 2):
        thresh = thresh / 100
        y_pred = (y_pred_labels > thresh).astype('int')
        score = f1_score(y_true, y_pred)
        #print(f"thresh: {thresh}, f1:{score}")
        if score > max_score:
            max_score = score
            max_thresh = thresh
        if y_pred.sum()/y_pred.shape[0] < 0.005:
            continue
    print(f"Optimum threshold: {max_thresh}, Score: {max_score}")
    return max_thresh, max_score

def calculate_optimum_threshold_05(y_true, y_pred_labels):
    max_thresh = 0.5
    max_score = 0
    y_pred = (y_pred_labels > max_thresh).astype('int')
    score = f1_score(y_true, y_pred)

    print(f"Optimum threshold: {max_thresh}, Score: {score}")
    return max_thresh, score


def overwrite_car_metrics(df):
    df.loc[df['model'] == "3 Series", 'model'] = 'BMW 3 Serisi'
    df.loc[df['model'] == "1 Series", 'model'] = 'BMW 1 Serisi'
    df.loc[df['model'] == "5 Series", 'model'] = 'BMW 5 Serisi'
    df.loc[df['model'] == "7 Series", 'model'] = 'BMW 7 Serisi'
    df.loc[df['model'] == "2 Series", 'model'] = 'BMW 2 Serisi'
    df.loc[df['model'] == "X1", 'model'] = 'BMW X1'
    df.loc[df['model'] == "X3", 'model'] = 'BMW X3'
    df.loc[df['model'] == "Hatch", 'model'] = 'MINI Hatch'
    df.loc[df['model'] == "Countryman", 'model'] = 'MINI Countryman'
    df.loc[df['model'] == "LR Range Rover Velar", 'model'] = 'Range Rover Velar'
    df.loc[df['model'] == "LR Range Rover Sport", 'model'] = 'Range Rover Sport'

    df.loc[df['make'] == "Land Rover", "make"] = 'LAND ROVER'
    df.loc[df['make'] == "Mini", "make"] = 'MINI'

    make_list = ['MINI', 'LAND ROVER', 'BMW']
    df = df[df['make'].isin(make_list)]
    df['year_to'] = df['year_to'].fillna(0).astype('int')
    df = df[df['year_to'] >= 2015]

    df["modeldef"] = np.nan
    df.loc[df['trim'] == "520i Steptronic", "modeldef"] = 'bmw 520i'
    df.loc[df['trim'] == "116d Steptronic", "modeldef"] = 'bmw 116d'
    df.loc[df['trim'] == "730Li Steptronic", "modeldef"] = 'bmw 730li'
    df.loc[df['id_trim'] == 6877, "modeldef"] = 'bmw x1 sdrive18i'
    df.loc[df['id_trim'] == 6340, "modeldef"] = 'bmw m550d xdrive'
    df.loc[df['trim'] == "320i Steptronic", "modeldef"] = 'bmw 320i'
    df.loc[df['id_trim'] == 39555, "modeldef"] = 'mini cooper d 5 kapi'
    df.loc[df['id_trim'] == 6883, "modeldef"] = 'bmw x1 sdrive16d'
    df.loc[df['trim'] == "530i xDrive Steptronic", "modeldef"] = 'bmw 530i xdrive'
    df.loc[df['trim'] == "520d xDrive Steptronic", "modeldef"] = 'bmw 520d xdrive'
    df.loc[df['id_trim'] == 6980, "modeldef"] = 'bmw x3 sdrive20i'
    df.loc[df['id_trim'] == 39525, "modeldef"] = 'mini cooper countryman all4'
    df.loc[df['trim'] == "530i Steptronic", "modeldef"] = 'bmw 530i'
    df.loc[df['trim'] == "520d Steptronic", "modeldef"] = 'bmw 520d'
    df.loc[(df['trim'] == "218d Steptronic") & (df['body_type'] == "Coupe"), "modeldef"] = 'bmw 216d gran coupe'
    df.loc[df['id_trim'] == 39556, "modeldef"] = 'mini cooper 5 kapi'
    df.loc[df['id_trim'] == 39526, "modeldef"] = 'mini cooper countryman'
    df.loc[df['trim'] == "730i Steptronic", "modeldef"] = 'bmw 730i'
    df.loc[df['id_trim'] == 39542, "modeldef"] = 'mini one d countryman'
    df.loc[df['id_trim'] == 4455, "modeldef"] = 'bmw 218i gran coupe'
    df.loc[df['trim'] == "730Ld xDrive Steptronic", "modeldef"] = 'bmw 730ld xdrive'
    df.loc[df['trim'] == "740Ld xDrive Steptronic", "modeldef"] = 'bmw 740ld xdrive'
    df.loc[df['trim'] == "xDrive20d Steptronic", "modeldef"] = 'bmw x3 xdrive20d'
    df.loc[df['trim'] == "xDrive20d AT", "modeldef"] = 'bmw x1 xdrive20d'
    df.loc[df['trim'] == "740Le xDrive Steptronic", "modeldef"] = 'bmw 740le xdrive iperformance'
    df.loc[df['trim'] == "bmw 725Ld Steptronic", "modeldef"] = 'bmw 725d'
    df.loc[df['trim'] == "bmw 725Ld Steptronic", "modeldef"] = 'bmw 725ld'
    df.loc[df['trim'] == "M235i xDrive Steptronic", "modeldef"] = 'bmw m235i xdrive gran coupe'
    df.loc[df['trim'] == "750Li xDrive Steptronic", "modeldef"] = 'bmw 750li xdrive'
    df.loc[df['trim'] == "750i xDrive Steptronic", "modeldef"] = 'bmw 750i xdrive'
    df.loc[df['trim'] == "730d xDrive Steptronic", "modeldef"] = 'bmw 730d xdrive'
    df.loc[df['trim'] == "M40i Steptronic", "modeldef"] = 'bmw x3 m40i'
    df.loc[df['trim'] == "xDrive25d Steptronic", "modeldef"] = 'bmw x1 xdrive25d'
    df.loc[df['trim'] == "750Li xDrive Steptronic", "modeldef"] = 'bmw 750li xdrive'
    df.loc[df['id_trim'] == 39540, "modeldef"] = 'mini john cooper works countryman'
    df.loc[df['trim'] == "740d xDrive Steptronic", "modeldef"] = 'bmw 740d xdrive'
    df.loc[df['trim'] == "740e Steptronic", "modeldef"] = 'bmw 740e iperformance'
    df.loc[df['trim'] == "745Le xDrive Steptronic", "modeldef"] = 'bmw 745le xdrive'
    df.loc[df['trim'] == "530d xDrive Steptronic", "modeldef"] = 'bmw 530d xdrive'
    df.loc[df['trim'] == "530d Steptronic", "modeldef"] = 'bmw 530d'
    df.loc[df['trim'] == "750d xDrive Steptronic", "modeldef"] = 'bmw 750d xdrive'
    df.loc[df['trim'] == "750Ld xDrive Steptronic", "modeldef"] = 'bmw 750ld xdrive'
    df.loc[df['trim'] == "750Le xDrive Steptronic", "modeldef"] = 'bmw 750le xdrive'
    df.loc[df['trim'] == "760Li xDrive Steptronic", "modeldef"] = 'bmw m760li xdrive'
    df.loc[df['trim'] == "530e iPerformance Steptronic", "modeldef"] = 'bmw 530e'
    df.loc[df['trim'] == "330e Steptronic", "modeldef"] = 'bmw 330e xdrive'
    df.loc[df['id_trim'] == 4348, "modeldef"] = 'bmw 116i'
    df.loc[df['trim'] == "118i Steptronic", "modeldef"] = 'bmw 118i'
    df.loc[df['trim'] == "540i xDrive Steptronic", "modeldef"] = 'bmw 540i xdrive'
    df.loc[df['id_trim'] == 4345, "modeldef"] = 'bmw 128ti'


    df.loc[df['id_trim'] == 32077, "modeldef"] = 'range rover sport 2.0 p300'
    df.loc[df['id_trim'] == 32078, "modeldef"] = 'range rover sport 2.0 sd4 240hp awd'
    df.loc[df['id_trim'] == 32079, "modeldef"] = 'range rover sport 3.0 sdv6'
    df.loc[df['id_trim'] == 32087, "modeldef"] = 'range rover velar 2.0 si4 250hp awd'
    df.loc[df['id_trim'] == 32088, "modeldef"] = 'range rover velar 2.0 d180'
    df.loc[df['id_trim'] == 32089, "modeldef"] = 'range rover velar 240hp awd'
    df.loc[df['id_trim'] == 32091, "modeldef"] = 'range rover velar 2.0 si4 300hp awd'

    df = df.append(df[df["id_trim"] == 32087], ignore_index=True)
    df = df.append(df[df["id_trim"] == 32088], ignore_index=True)
    df = df.append(df[df["id_trim"] == 32088], ignore_index=True)
    df = df.append(df[df["id_trim"] == 32089], ignore_index=True)

    df.iloc[1593, df.columns.get_loc('modeldef')] = 'range rover velar 2.0 d204'
    df.iloc[1594, df.columns.get_loc('modeldef')] = 'range rover velar 2.0 td4 180hp awd'
    df.iloc[1596, df.columns.get_loc('modeldef')] = 'range rover velar 2.0 d240'
    df.iloc[1592, df.columns.get_loc('modeldef')] = 'range rover velar 2.0 p250'

    df = df[df['modeldef'].notna()]

    return df

