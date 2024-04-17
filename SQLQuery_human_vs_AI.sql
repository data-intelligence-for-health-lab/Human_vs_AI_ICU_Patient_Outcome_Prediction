---------------Mastertable------------------
SELECT A."PATIENT_ID", A."ADMISSION_ID", A."AGE_AT_ICU_ADMIT", P."GENDER",
ROUND(EXTRACT(EPOCH FROM (A."ICU_DISCH_DATETIME" - A."ICU_ADMIT_DATETIME")) / (24 * 60 * 60), 1) AS ICU_LOS,
ROUND(EXTRACT(EPOCH FROM (A."HOSP_DISCH_DATETIME" - A."HOSP_ADMIT_DATETIME")) / (24 * 60 * 60), 1) AS HOSP_LOS,
A."ICU_EXPIRE_FLAG", A."HOSP_EXPIRE_FLAG", A."30_DAYS_EXPIRE_FLAG", A."DELIRIUM_FLAG"
FROM public."ADMISSIONS" A
JOIN public."PATIENTS" P ON A."PATIENT_ID" = P."PATIENT_ID"
WHERE "ICU_LOS_24H_FLAG" = 1
ORDER BY A."PATIENT_ID";

--------------------Intervention_24h----------------------------
DROP TABLE IF EXISTS "lu.kuang".INTERVENTIONS_24hr;
CREATE TABLE "lu.kuang".INTERVENTIONS_24hr AS
SELECT
    I."ADMISSION_ID",
    I."ITEM_ID",
    I."VALUE_NUM",
    I."VALUE_CHAR",
    I."START_DATETIME",
    A."ICU_ADMIT_DATETIME",
    IT."LABEL",
    EXTRACT(epoch FROM I."START_DATETIME" - A."ICU_ADMIT_DATETIME") / 3600 AS time_difference_hours
FROM
    public."INTERVENTIONS" I
JOIN
    public."ADMISSIONS" A ON A."ADMISSION_ID" = I."ADMISSION_ID"
JOIN
    public."ITEMS" IT ON IT."ITEM_ID" = I."ITEM_ID"
WHERE
EXTRACT(epoch FROM I."START_DATETIME" - A."ICU_ADMIT_DATETIME") / 3600 < 24;

----------------FIRST 24 MEASUREMENT-------------------- 
SELECT 
    M."ADMISSION_ID", 
    M."ITEM_ID", 
    M."VALUE_NUM", 
    M."VALUE_CHAR", 
    M.sDATETIME", 
    A."ICU_ADMIT_DATETIME", 
    IT."LABEL", 
EXTRACT(epoch FROM M."DATETIME" - A."ICU_ADMIT_DATETIME") / 3600 AS time_difference_hours 
FROM 
    public."MEASUREMENTS" M 
JOIN 
    public."ADMISSIONS" A ON A."ADMISSION_ID" = M."ADMISSION_ID" 
JOIN 
    public."ITEMS" IT ON IT."ITEM_ID" = M."ITEM_ID" 
WHERE 
    EXTRACT(epoch FROM M."DATETIME" - A."ICU_ADMIT_DATETIME") / 3600 < 24;

-------------FIRST 24 Prescription-------------------
SELECT 
    P."ADMISSION_ID", 
    P."ITEM_ID", 
    P."VALUE_NUM", 
P."VALUE_UOM", 
    P."RATE_NUM", 
P."RATE_UOM", 
    P."START_DATETIME", 
    A."ICU_ADMIT_DATETIME", 
    IT."LABEL", 
    EXTRACT(epoch FROM P."START_DATETIME" - A."ICU_ADMIT_DATETIME") / 3600 AS time_difference_hours 
FROM 
    public."PRESCRIPTIONS" P 
JOIN 
    public."ADMISSIONS" A ON A."ADMISSION_ID" = P."ADMISSION_ID" 
JOIN 
    public."ITEMS" IT ON IT."ITEM_ID" = P."ITEM_ID" 
WHERE 
    EXTRACT(epoch FROM P."START_DATETIME"- A."ICU_ADMIT_DATETIME") / 3600 < 24; 

----------------Measurements_value_char > 30%----------------------
SELECT "ITEM_ID", "LABEL", COUNT(DISTINCT "ADMISSION_ID") AS COUNT
FROM "lu.kuang".measurements_24hr
WHERE "VALUE_CHAR" <> ''
GROUP BY "ITEM_ID", "LABEL"
HAVING COUNT(DISTINCT "ADMISSION_ID") > ((SELECT COUNT(DISTINCT "ADMISSION_ID") FROM "lu.kuang".measurements_24hr) * 0.3)
ORDER BY "ITEM_ID", COUNT DESC;

----------------Interventions_value_char > 30%----------------------
SELECT "ITEM_ID", "LABEL", COUNT(DISTINCT "ADMISSION_ID") AS COUNT
FROM "lu.kuang".interventions_24hr
WHERE "VALUE_CHAR" <> ''
GROUP BY "ITEM_ID", "LABEL"
HAVING COUNT(DISTINCT "ADMISSION_ID") > ((SELECT COUNT(DISTINCT "ADMISSION_ID") FROM "lu.kuang".interventions_24hr) * 0.3)
ORDER BY "ITEM_ID", COUNT DESC;

-----------------Interventions_value_char--------------------------
SELECT "ITEM_ID", "LABEL", "VALUE_CHAR", COUNT("ADMISSION_ID") AS COUNT
FROM "lu.kuang".interventions_24hr
WHERE "ITEM_ID" IN ('I311',
'I318',
'I323',
'I327',
'I331')
GROUP BY "ITEM_ID", "LABEL", "VALUE_CHAR"
ORDER BY "ITEM_ID";

-----------------Mesurements_var_char----------------------------------
SELECT "ADMISSION_ID","ITEM_ID", "LABEL", "VALUE_CHAR", COUNT("ADMISSION_ID") AS COUNT
FROM "lu.kuang".measurements_24hr
WHERE "ITEM_ID" IN ('I004', 'I016', 'I019')
GROUP BY "ADMISSION_ID","ITEM_ID", "LABEL", "VALUE_CHAR"
ORDER BY "ADMISSION_ID";

