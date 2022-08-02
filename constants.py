class C:
    run_optuna = False
    run_fast_trial = False
    run_lofo = False
    run_city = False
    run_adv = False
    use_nfolds = True
    optuna_trials = 100
    check_val_results = True
    use_scaler = False
    run_imputer = False
    plot_importance = True
    run_perm_importance = True
    run_feature_interactions = False
    trial = 7
    model_type = "cat"
    read_ready_files = True
    validation_type = "stratified"