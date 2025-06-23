CREATE TABLE "lu.kuang".prospective_prescriptions (
    prescriptions_id VARCHAR,
    admission_id VARCHAR,
    patient_id VARCHAR,
    item_id VARCHAR,
    start_datetime TIMESTAMP,
    end_datetime TIMESTAMP,
    delta_time INTEGER,
    delta_uom VARCHAR,
    value_num NUMERIC,
    value_uom VARCHAR,
    rate_num NUMERIC,
    rate_uom VARCHAR,
    site VARCHAR,
    mrn VARCHAR,
    icu_admit TIMESTAMP,
    label VARCHAR
);