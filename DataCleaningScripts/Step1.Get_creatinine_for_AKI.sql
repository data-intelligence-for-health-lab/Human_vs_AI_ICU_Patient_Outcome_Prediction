<<<<<<< HEAD:SQL Scripts/AKI_rawdata.sql
--This table prepares the raw data for AKI flag to be derived in AKI.py
=======
--This table prepares the raw data (exported as AKI_Cr.csv)for AKI flag to be derived in AKI.py
>>>>>>> f42f53877a7cffcc286d980785572b82a861908f:DataCleaningScripts/Step1.Get_creatinine_for_AKI.sql
SELECT 
    M."ADMISSION_ID", 
    M."ITEM_ID", 
    M."VALUE_NUM", 
    M."VALUE_CHAR", 
    M."DATETIME", 
    A."ICU_ADMIT_DATETIME", 
    IT."LABEL", 
EXTRACT(epoch FROM M."DATETIME" - A."ICU_ADMIT_DATETIME") / 3600 AS measurement_hour 
FROM 
    public."MEASUREMENTS" M 
JOIN 
    public."ADMISSIONS" A ON A."ADMISSION_ID" = M."ADMISSION_ID" 
JOIN 
    public."ITEMS" IT ON IT."ITEM_ID" = M."ITEM_ID" 
WHERE 
    M."ITEM_ID" = 'I056' --creatinine