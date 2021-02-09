query_longitudinal = """
SELECT subject_id, hadm_id, icustay_id,
        DATE_PART('day', icutime) AS ic_t,

        -- VITALS
        MAX(HeartRate)  AS HeartRateHigh,
        AVG(SysBP)      AS SysBP,
        AVG(DiasBP)     AS DiasBP,
        AVG(MeanBP)     AS MeanBP,
        AVG(RespRate)   AS RespRate,
        AVG(Temp)       AS Temperature,
        AVG(GlucoseChart) AS GlucoseChart,
        AVG(SpO2)       AS SpO2,
        AVG(FiO2)       AS FiO2,

        -- LAB VALUES
        AVG(Bicarbonate) AS Bicarbonate,
        AVG(Creatinine)  AS Creatinine,
        AVG(Chloride)    AS Chloride,
        AVG(GlucoseLab)  AS GlucoseLab,
        AVG(Hematocrit)  AS Hematocrit, 
        AVG(Hemoglobin)  AS Hemoglobin,
        AVG(Platelet)    AS Platelet,
        AVG(Potassium)   AS Potassium,  
        AVG(PTT)         AS PTT,
        AVG(INR)         AS INR,
        AVG(PT)          AS PT,
        AVG(Sodium)      AS Sodium,
        AVG(BUN)         AS BUN,
        AVG(WBC)         AS WBC
        -- AVG(AnionGap)    AS AnionGap, 
        -- AVG(Albumin)     AS Albumin,
        -- AVG(Bands)       AS Bands, 
        -- AVG(Bilirubin)   AS Bilirubin, 
        -- AVG(Lactate)     AS Lactate

FROM (SELECT * FROM easychart ec FULL OUTER JOIN easylabs el USING (subject_id, hadm_id, icustay_id, icutime)) easymerge

GROUP BY subject_id, hadm_id, icustay_id, ic_t
ORDER BY icustay_id, ic_t
    """
