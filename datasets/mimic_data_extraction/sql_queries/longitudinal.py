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
        AVG(WBC)         AS WBC,
        -- AVG(AnionGap)    AS AnionGap, 
        -- AVG(Albumin)     AS Albumin,
        -- AVG(Bands)       AS Bands, 
        -- AVG(Bilirubin)   AS Bilirubin, 
        -- AVG(Lactate)     AS Lactate

        -- NEW
        AVG(TidalVolume)              AS TidalVolume,
        AVG(MinuteVolume)             AS MinuteVolume,
        AVG(IERatio)                  AS IERatio,
        AVG(PEEP)                     AS PEEP,
        AVG(SetPressureOnVentilator)  AS SetPressureOnVentilator,
        AVG(PeakInspPressure)         AS PeakInspPressure,
        AVG(MeanAirwayPressure)       AS MeanAirwayPressure,
        AVG(PlateauPressure)          AS PlateauPressure,
        AVG(PaO2)                     AS PaO2,
        AVG(PaCO2)                    AS PaCO2,
        AVG(pH)                       AS pH,
        AVG(LacticAcid)               AS LacticAcid,
        AVG(Leukocytes)               AS Leukocytes,
        AVG(CRP)                      AS CRP,
        
        AVG(GCSVerbal)                AS GCSVerbal,
        AVG(GCSMotor)                 AS GCSMotor,
        AVG(GCSEyes)                  AS GCSEyes,

        -- TEXT
        MIN(VentilationModeCV)        AS VentilationModeCV,
        MIN(VentilationModeMV)        AS VentilationModeMV

FROM (SELECT * FROM easychart ec FULL OUTER JOIN easylabs el USING (subject_id, hadm_id, icustay_id, icutime)) easymerge

GROUP BY subject_id, hadm_id, icustay_id, ic_t
ORDER BY icustay_id, ic_t
    """
