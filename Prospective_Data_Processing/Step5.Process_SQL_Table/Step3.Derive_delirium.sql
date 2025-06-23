CREATE TABLE DELIRIUM_STATUS AS (
    SELECT 
        M."ADMISSION_ID",
        MAX(CASE
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) <= 24 
                 AND M."VALUE_NUM" >= 4 THEN 1
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) <= 24 
                 AND M."VALUE_NUM" < 4 THEN 0
            ELSE NULL
        END) AS DELIRIUM_STATUS_24H,
        
        MAX(CASE
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) > 24 
                 AND EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) <= 48 
                 AND M."VALUE_NUM" >= 4 THEN 1
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) > 24 
                 AND EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) <= 48 
                 AND M."VALUE_NUM" < 4 THEN 0
            ELSE NULL
        END) AS DELIRIUM_STATUS_24_48H,
        
        MAX(CASE
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) > 48 
                 AND M."VALUE_NUM" >= 4 THEN 1
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) > 48 
                 AND M."VALUE_NUM" < 4 THEN 0
            ELSE NULL
        END) AS DELIRIUM_STATUS_48H_PLUS

    FROM 
        public."ADMISSIONS" A
        JOIN public."MEASUREMENTS" M ON A."ADMISSION_ID" = M."ADMISSION_ID"
    WHERE 
        "ICU_LOS_24H_FLAG" = 1
        AND M."ITEM_ID" = 'I007' -- ICDSC SCORE
    GROUP BY 
        M."ADMISSION_ID"