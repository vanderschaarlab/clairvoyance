from utils import PipelineComposer
from preprocessing import FilterNegative, OneHotEncoder, Normalizer
from imputation import Imputation
from feature_selection import FeatureSelection
from prediction import prediction
from uncertainty import uncertainty
from interpretation import interpretation
from active_sensing import active_sensing

from .interface import Results


def timeseries_prediction_pl(problem_maker, extra_settings, dataset_training, dataset_testing) -> Results:

    # --- Preprocessing.

    # (1) filter out negative values
    negative_filter = FilterNegative()
    # (2) one-hot encode categorical features
    one_hot_encoding = "admission_type"
    onehot_encoder = OneHotEncoder(one_hot_encoding_features=[one_hot_encoding])
    # (3) Normalize features: 3 options (minmax, standard, none)
    normalization = "minmax"
    normalizer = Normalizer(normalization)

    # Data preprocessing
    filter_pipeline = PipelineComposer(negative_filter, onehot_encoder, normalizer)

    dataset_training = filter_pipeline.fit_transform(dataset_training)
    dataset_testing = filter_pipeline.transform(dataset_testing)

    print("Finish preprocessing.")

    # --- Problem maker fit_transform step.
    dataset_training = problem_maker.fit_transform(dataset_training)
    dataset_testing = problem_maker.fit_transform(dataset_testing)

    # --- Imputation.

    # Set imputation models
    static_imputation_model = "median"
    temporal_imputation_model = "median"

    # Impute the missing data
    static_imputation = Imputation(imputation_model_name=static_imputation_model, data_type="static")
    temporal_imputation = Imputation(imputation_model_name=temporal_imputation_model, data_type="temporal")

    imputation_pipeline = PipelineComposer(static_imputation, temporal_imputation)

    dataset_training = imputation_pipeline.fit_transform(dataset_training)
    dataset_testing = imputation_pipeline.transform(dataset_testing)

    print("Finish imputation.")

    # --- Feature selection.

    # Set feature selection parameters
    static_feature_selection_model = None
    temporal_feature_selection_model = None
    static_feature_selection_number = None
    temporal_feature_selection_number = None

    # Select relevant features
    static_feature_selection = FeatureSelection(
        feature_selection_model_name=static_feature_selection_model,
        feature_type="static",
        feature_number=static_feature_selection_number,
        task=extra_settings.task,
        metric_name=extra_settings.metric_name,
        metric_parameters=extra_settings.metric_parameters,
    )

    temporal_feature_selection = FeatureSelection(
        feature_selection_model_name=temporal_feature_selection_model,
        feature_type="temporal",
        feature_number=temporal_feature_selection_number,
        task=extra_settings.task,
        metric_name=extra_settings.metric_name,
        metric_parameters=extra_settings.metric_parameters,
    )

    feature_selection_pipeline = PipelineComposer(static_feature_selection, temporal_feature_selection)

    dataset_training = feature_selection_pipeline.fit_transform(dataset_training)
    dataset_testing = feature_selection_pipeline.transform(dataset_testing)

    print("Finish feature selection.")

    # --- Prediction.

    # Set up validation for early stopping and best model saving
    dataset_training.train_val_test_split(prob_val=0.2, prob_test=0.0)

    # Train the predictive model
    pred_class = prediction(extra_settings.model_name, extra_settings.model_parameters, extra_settings.task)
    pred_class.fit(dataset_training)
    # Return the predictions on the testing set
    test_y_hat = pred_class.predict(dataset_testing)

    print("Finish predictor model training and testing.")

    # --- Uncertainty.

    # Set uncertainty model
    uncertainty_model_name = "ensemble"

    # Train uncertainty model
    uncertainty_model = uncertainty(
        uncertainty_model_name, extra_settings.model_parameters, pred_class, extra_settings.task
    )
    uncertainty_model.fit(dataset_training)
    # Return uncertainty of the trained predictive model
    test_ci_hat = uncertainty_model.predict(dataset_testing)

    print("Finish uncertainty estimation")

    # --- Interpretation.

    # Set interpretation model
    interpretation_model_name = "tinvase"

    # Train interpretation model
    interpretor = interpretation(
        interpretation_model_name, extra_settings.model_parameters, pred_class, extra_settings.task
    )
    interpretor.fit(dataset_training)
    # Return instance-wise temporal and static feature importance
    test_s_hat = interpretor.predict(dataset_testing)

    print("Finish model interpretation")

    return Results(
        test_y_hat=test_y_hat,
        test_ci_hat=test_ci_hat,
        test_s_hat=test_s_hat,
        dataset_training=dataset_training,
        dataset_testing=dataset_testing,
    )


def active_sensing_pl(problem_maker, extra_settings, dataset_training, dataset_testing) -> Results:

    # --- Preprocessing.

    # (1) filter out negative values
    negative_filter = FilterNegative()
    # (2) one-hot encode categorical features
    one_hot_encoding = "admission_type"
    onehot_encoder = OneHotEncoder(one_hot_encoding_features=[one_hot_encoding])
    # (3) Normalize features: 3 options (minmax, standard, none)
    normalization = "minmax"
    normalizer = Normalizer(normalization)

    # Data preprocessing
    filter_pipeline = PipelineComposer(negative_filter, onehot_encoder, normalizer)

    dataset_training = filter_pipeline.fit_transform(dataset_training)
    dataset_testing = filter_pipeline.transform(dataset_testing)

    print("Finish preprocessing.")

    # --- Problem maker fit_transform step.
    dataset_training = problem_maker.fit_transform(dataset_training)
    dataset_testing = problem_maker.fit_transform(dataset_testing)

    # --- Imputation.

    # Set imputation models
    static_imputation_model = "median"
    temporal_imputation_model = "median"

    # Impute the missing data
    static_imputation = Imputation(imputation_model_name=static_imputation_model, data_type="static")
    temporal_imputation = Imputation(imputation_model_name=temporal_imputation_model, data_type="temporal")

    imputation_pipeline = PipelineComposer(static_imputation, temporal_imputation)

    dataset_training = imputation_pipeline.fit_transform(dataset_training)
    dataset_testing = imputation_pipeline.transform(dataset_testing)

    print("Finish imputation.")

    # --- Feature selection.

    # Set feature selection parameters
    static_feature_selection_model = None
    temporal_feature_selection_model = None
    static_feature_selection_number = None
    temporal_feature_selection_number = None

    # Select relevant features
    static_feature_selection = FeatureSelection(
        feature_selection_model_name=static_feature_selection_model,
        feature_type="static",
        feature_number=static_feature_selection_number,
        task=extra_settings.task,
        metric_name=extra_settings.metric_name,
        metric_parameters=extra_settings.metric_parameters,
    )

    temporal_feature_selection = FeatureSelection(
        feature_selection_model_name=temporal_feature_selection_model,
        feature_type="temporal",
        feature_number=temporal_feature_selection_number,
        task=extra_settings.task,
        metric_name=extra_settings.metric_name,
        metric_parameters=extra_settings.metric_parameters,
    )

    feature_selection_pipeline = PipelineComposer(static_feature_selection, temporal_feature_selection)

    dataset_training = feature_selection_pipeline.fit_transform(dataset_training)
    dataset_testing = feature_selection_pipeline.transform(dataset_testing)

    print("Finish feature selection.")

    # --- Active Sensing.

    # Set up validation for early stopping and best model saving
    dataset_training.train_val_test_split(prob_val=0.2, prob_test=0.0)

    # Train the original predictive model
    active_sensing_class = active_sensing(
        extra_settings.model_name, extra_settings.model_parameters, extra_settings.task
    )
    active_sensing_class.fit(dataset_training)
    # Return the observation recommendations on the testing set
    test_s_hat = active_sensing_class.predict(dataset_testing)

    print("Finish active sensing model training and testing.")

    return Results(
        test_y_hat=None,
        test_ci_hat=None,
        test_s_hat=test_s_hat,
        dataset_training=dataset_training,
        dataset_testing=dataset_testing,
    )


def ite_pl(problem_maker, extra_settings, dataset_training, dataset_testing) -> Results:
    # TODO: This
    pass
