CREATE TABLE aggregated_delirium_data AS
SELECT
  sub."ADMISSION_ID",
  BOOL_OR(sub."<24") AS "<24",
  BOOL_OR(sub."24-48") AS "24-48",
  BOOL_OR(sub.">=48") AS ">=48",
  MAX(sub."DELIRIUM_FLAG") AS "DELIRIUM_FLAG"  -- Aggregating DELIRIUM_FLAG
FROM
  (
    SELECT 
      m."ADMISSION_ID",
      a."DELIRIUM_FLAG",  -- Select DELIRIUM_FLAG from the ADMISSIONS table
      CASE 
        WHEN m."VALUE_NUM" < 4 THEN FALSE
        ELSE CASE 
          WHEN EXTRACT(EPOCH FROM (m."DATETIME" - a."HOSP_ADMIT_DATETIME")) / 3600 < 24 THEN TRUE 
          ELSE FALSE 
        END
      END AS "<24",
      CASE 
        WHEN m."VALUE_NUM" < 4 THEN FALSE
        ELSE CASE 
          WHEN EXTRACT(EPOCH FROM (m."DATETIME" - a."HOSP_ADMIT_DATETIME")) / 3600 >= 24 
               AND EXTRACT(EPOCH FROM (m."DATETIME" - a."HOSP_ADMIT_DATETIME")) / 3600 < 48 THEN TRUE 
          ELSE FALSE 
        END
      END AS "24-48",
      CASE 
        WHEN m."VALUE_NUM" < 4 THEN FALSE
        ELSE CASE 
          WHEN EXTRACT(EPOCH FROM (m."DATETIME" - a."HOSP_ADMIT_DATETIME")) / 3600 >= 48 THEN TRUE 
          ELSE FALSE 
        END
      END AS ">=48"
    FROM 
      public."MEASUREMENTS" AS m
    JOIN 
      public."ADMISSIONS" AS a ON m."ADMISSION_ID" = a."ADMISSION_ID"
    WHERE 
      m."ITEM_ID" = 'I007'
  ) sub
GROUP BY 
  sub."ADMISSION_ID";