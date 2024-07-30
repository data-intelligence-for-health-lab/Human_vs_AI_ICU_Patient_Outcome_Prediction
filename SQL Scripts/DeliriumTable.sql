SELECT 
        M."ADMISSION_ID",
        -- Delirium flag for <= 24 hours
        MAX(CASE
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) <= 24 THEN 1
            ELSE 0
        END) AS DELIRIUM_FLAG_24H,
        -- Delirium flag for > 24 hours and <= 48 hours
        MAX(CASE
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) > 24 
             AND EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) <= 48 THEN 1
            ELSE 0
        END) AS DELIRIUM_FLAG_24_48H,
        -- Delirium flag for > 48 hours
        MAX(CASE
            WHEN EXTRACT(EPOCH FROM (M."DATETIME" - A."ICU_ADMIT_DATETIME")) / (60 * 60) > 48 THEN 1
            ELSE 0
        END) AS DELIRIUM_FLAG_48H_PLUS
    FROM 
        public."ADMISSIONS" A
        JOIN public."MEASUREMENTS" M ON A."ADMISSION_ID" = M."ADMISSION_ID"
    WHERE 
        "ICU_LOS_24H_FLAG" = 1
        AND M."ITEM_ID" = 'I007'
        AND M."VALUE_NUM" >= 4
    GROUP BY 
        M."ADMISSION_ID"