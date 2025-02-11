CREATE TABLE "lu.kuang".final_table AS
SELECT
    *,
    CASE
        WHEN "Delirium_24h" IS TRUE THEN NULL
        WHEN "Delirium_24h" IS NULL THEN NULL
        WHEN "Delirium_24_48h" IS TRUE OR "Delirium_48h" IS TRUE THEN TRUE
        ELSE FALSE
    END AS delirium_label,
    CASE
        WHEN "aki_status_24h" IS TRUE THEN NULL
        WHEN "aki_status_24h" IS NULL THEN NULL
        WHEN "aki_status_24_48h" IS TRUE OR "aki_status_48h" IS TRUE THEN TRUE
        ELSE FALSE
    END AS aki_label
FROM
    "lu.kuang".final_mastertable_merged;