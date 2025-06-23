CREATE TABLE prospective_interventions (
    intervention_id TEXT,
    admission_id TEXT,
    patient_id TEXT,
    item_id TEXT,
    start_datetime TIMESTAMP,
    end_datetime TIMESTAMP,
    value_num FLOAT,
    value_uom TEXT,
    value_char TEXT,
    timeduration_mi FLOAT,
    orderdosevalue_num FLOAT,
    orderdosevalue_unit TEXT,
    site TEXT,
    icu_admit TIMESTAMP
);