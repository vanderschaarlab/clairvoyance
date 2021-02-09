DROP TABLE IF EXISTS cohort CASCADE;
CREATE UNLOGGED TABLE cohort AS

SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime, ie.outtime, 
    CASE WHEN icustay_id%10 < 8 THEN 0
         WHEN icustay_id%10 = 8 THEN 1
         ELSE 2 END AS group
FROM icustays ie

-- Extract ages
INNER JOIN (SELECT a.hadm_id,
            MIN(ROUND((cast(a.admittime as date) - cast(p.dob as date)) / 365.242,2)) AS first_admit_age
            FROM patients p
            INNER JOIN admissions a USING (subject_id)
            GROUP BY hadm_id) age
USING (hadm_id)

-- Inclusion criteria
WHERE age.first_admit_age > 1 -- Exclude neonates
  AND age.first_admit_age < 100 -- Exclude very old patients
  AND DATE_PART('day', ie.outtime - ie.intime) >= 3 -- Exclude short stays (<3 days)
  AND ie.dbsource = 'metavision'
;
-- Index
CREATE INDEX ch_index ON cohort (icustay_id)
