# Extract patients from MIMIC that have received antibiotics
# The patients on antibiotics will be identified by icustay_id and hadm_id.

# Used code from https://github.com/alistairewj/sepsis3-mimic/blob/master/appendix/defining-suspected-infection.ipynb


query_ab_ditems = """
with di as
(
  select
    di.*
    , case
      when lower(label) like '%' || lower('adoxa') || '%' then 1
      when lower(label) like '%' || lower('ala-tet') || '%' then 1
      when lower(label) like '%' || lower('alodox') || '%' then 1
      when lower(label) like '%' || lower('amikacin') || '%' then 1
      when lower(label) like '%' || lower('amikin') || '%' then 1
      when lower(label) like '%' || lower('amoxicillin') || '%' then 1
      when lower(label) like '%' || lower('amoxicillin%clavulanate') || '%' then 1
      when lower(label) like '%' || lower('clavulanate') || '%' then 1
      when lower(label) like '%' || lower('ampicillin') || '%' then 1
      when lower(label) like '%' || lower('augmentin') || '%' then 1
      when lower(label) like '%' || lower('avelox') || '%' then 1
      when lower(label) like '%' || lower('avidoxy') || '%' then 1
      when lower(label) like '%' || lower('azactam') || '%' then 1
      when lower(label) like '%' || lower('azithromycin') || '%' then 1
      when lower(label) like '%' || lower('aztreonam') || '%' then 1
      when lower(label) like '%' || lower('axetil') || '%' then 1
      when lower(label) like '%' || lower('bactocill') || '%' then 1
      when lower(label) like '%' || lower('bactrim') || '%' then 1
      when lower(label) like '%' || lower('bethkis') || '%' then 1
      when lower(label) like '%' || lower('biaxin') || '%' then 1
      when lower(label) like '%' || lower('bicillin l-a') || '%' then 1
      when lower(label) like '%' || lower('cayston') || '%' then 1
      when lower(label) like '%' || lower('cefazolin') || '%' then 1
      when lower(label) like '%' || lower('cedax') || '%' then 1
      when lower(label) like '%' || lower('cefoxitin') || '%' then 1
      when lower(label) like '%' || lower('ceftazidime') || '%' then 1
      when lower(label) like '%' || lower('cefaclor') || '%' then 1
      when lower(label) like '%' || lower('cefadroxil') || '%' then 1
      when lower(label) like '%' || lower('cefdinir') || '%' then 1
      when lower(label) like '%' || lower('cefditoren') || '%' then 1
      when lower(label) like '%' || lower('cefepime') || '%' then 1
      when lower(label) like '%' || lower('cefotetan') || '%' then 1
      when lower(label) like '%' || lower('cefotaxime') || '%' then 1
      when lower(label) like '%' || lower('cefpodoxime') || '%' then 1
      when lower(label) like '%' || lower('cefprozil') || '%' then 1
      when lower(label) like '%' || lower('ceftibuten') || '%' then 1
      when lower(label) like '%' || lower('ceftin') || '%' then 1
      when lower(label) like '%' || lower('cefuroxime ') || '%' then 1
      when lower(label) like '%' || lower('cefuroxime') || '%' then 1
      when lower(label) like '%' || lower('cephalexin') || '%' then 1
      when lower(label) like '%' || lower('chloramphenicol') || '%' then 1
      when lower(label) like '%' || lower('cipro') || '%' then 1
      when lower(label) like '%' || lower('ciprofloxacin') || '%' then 1
      when lower(label) like '%' || lower('claforan') || '%' then 1
      when lower(label) like '%' || lower('clarithromycin') || '%' then 1
      when lower(label) like '%' || lower('cleocin') || '%' then 1
      when lower(label) like '%' || lower('clindamycin') || '%' then 1
      when lower(label) like '%' || lower('cubicin') || '%' then 1
      when lower(label) like '%' || lower('dicloxacillin') || '%' then 1
      when lower(label) like '%' || lower('doryx') || '%' then 1
      when lower(label) like '%' || lower('doxycycline') || '%' then 1
      when lower(label) like '%' || lower('duricef') || '%' then 1
      when lower(label) like '%' || lower('dynacin') || '%' then 1
      when lower(label) like '%' || lower('ery-tab') || '%' then 1
      when lower(label) like '%' || lower('eryped') || '%' then 1
      when lower(label) like '%' || lower('eryc') || '%' then 1
      when lower(label) like '%' || lower('erythrocin') || '%' then 1
      when lower(label) like '%' || lower('erythromycin') || '%' then 1
      when lower(label) like '%' || lower('factive') || '%' then 1
      when lower(label) like '%' || lower('flagyl') || '%' then 1
      when lower(label) like '%' || lower('fortaz') || '%' then 1
      when lower(label) like '%' || lower('furadantin') || '%' then 1
      when lower(label) like '%' || lower('garamycin') || '%' then 1
      when lower(label) like '%' || lower('gentamicin') || '%' then 1
      when lower(label) like '%' || lower('kanamycin') || '%' then 1
      when lower(label) like '%' || lower('keflex') || '%' then 1
      when lower(label) like '%' || lower('ketek') || '%' then 1
      when lower(label) like '%' || lower('levaquin') || '%' then 1
      when lower(label) like '%' || lower('levofloxacin') || '%' then 1
      when lower(label) like '%' || lower('lincocin') || '%' then 1
      when lower(label) like '%' || lower('macrobid') || '%' then 1
      when lower(label) like '%' || lower('macrodantin') || '%' then 1
      when lower(label) like '%' || lower('maxipime') || '%' then 1
      when lower(label) like '%' || lower('mefoxin') || '%' then 1
      when lower(label) like '%' || lower('metronidazole') || '%' then 1
      when lower(label) like '%' || lower('minocin') || '%' then 1
      when lower(label) like '%' || lower('minocycline') || '%' then 1
      when lower(label) like '%' || lower('monodox') || '%' then 1
      when lower(label) like '%' || lower('monurol') || '%' then 1
      when lower(label) like '%' || lower('morgidox') || '%' then 1
      when lower(label) like '%' || lower('moxatag') || '%' then 1
      when lower(label) like '%' || lower('moxifloxacin') || '%' then 1
      when lower(label) like '%' || lower('myrac') || '%' then 1
      when lower(label) like '%' || lower('nafcillin sodium') || '%' then 1
      when lower(label) like '%' || lower('nicazel doxy 30') || '%' then 1
      when lower(label) like '%' || lower('nitrofurantoin') || '%' then 1
      when lower(label) like '%' || lower('noroxin') || '%' then 1
      when lower(label) like '%' || lower('ocudox') || '%' then 1
      when lower(label) like '%' || lower('ofloxacin') || '%' then 1
      when lower(label) like '%' || lower('omnicef') || '%' then 1
      when lower(label) like '%' || lower('oracea') || '%' then 1
      when lower(label) like '%' || lower('oraxyl') || '%' then 1
      when lower(label) like '%' || lower('oxacillin') || '%' then 1
      when lower(label) like '%' || lower('pc pen vk') || '%' then 1
      when lower(label) like '%' || lower('pce dispertab') || '%' then 1
      when lower(label) like '%' || lower('panixine') || '%' then 1
      when lower(label) like '%' || lower('pediazole') || '%' then 1
      when lower(label) like '%' || lower('penicillin') || '%' then 1
      when lower(label) like '%' || lower('periostat') || '%' then 1
      when lower(label) like '%' || lower('pfizerpen') || '%' then 1
      when lower(label) like '%' || lower('piperacillin') || '%' then 1
      when lower(label) like '%' || lower('tazobactam') || '%' then 1
      when lower(label) like '%' || lower('primsol') || '%' then 1
      when lower(label) like '%' || lower('proquin') || '%' then 1
      when lower(label) like '%' || lower('raniclor') || '%' then 1
      when lower(label) like '%' || lower('rifadin') || '%' then 1
      when lower(label) like '%' || lower('rifampin') || '%' then 1
      when lower(label) like '%' || lower('rocephin') || '%' then 1
      when lower(label) like '%' || lower('smz-tmp') || '%' then 1
      when lower(label) like '%' || lower('septra') || '%' then 1
      when lower(label) like '%' || lower('septra ds') || '%' then 1
      when lower(label) like '%' || lower('septra') || '%' then 1
      when lower(label) like '%' || lower('solodyn') || '%' then 1
      when lower(label) like '%' || lower('spectracef') || '%' then 1
      when lower(label) like '%' || lower('streptomycin sulfate') || '%' then 1
      when lower(label) like '%' || lower('sulfadiazine') || '%' then 1
      when lower(label) like '%' || lower('sulfamethoxazole') || '%' then 1
      when lower(label) like '%' || lower('trimethoprim') || '%' then 1
      when lower(label) like '%' || lower('sulfatrim') || '%' then 1
      when lower(label) like '%' || lower('sulfisoxazole') || '%' then 1
      when lower(label) like '%' || lower('suprax') || '%' then 1
      when lower(label) like '%' || lower('synercid') || '%' then 1
      when lower(label) like '%' || lower('tazicef') || '%' then 1
      when lower(label) like '%' || lower('tetracycline') || '%' then 1
      when lower(label) like '%' || lower('timentin') || '%' then 1
      when lower(label) like '%' || lower('tobi') || '%' then 1
      when lower(label) like '%' || lower('tobramycin') || '%' then 1
      when lower(label) like '%' || lower('trimethoprim') || '%' then 1
      when lower(label) like '%' || lower('unasyn') || '%' then 1
      when lower(label) like '%' || lower('vancocin') || '%' then 1
      when lower(label) like '%' || lower('vancomycin') || '%' then 1
      when lower(label) like '%' || lower('vantin') || '%' then 1
      when lower(label) like '%' || lower('vibativ') || '%' then 1
      when lower(label) like '%' || lower('vibra-tabs') || '%' then 1
      when lower(label) like '%' || lower('vibramycin') || '%' then 1
      when lower(label) like '%' || lower('zinacef') || '%' then 1
      when lower(label) like '%' || lower('zithromax') || '%' then 1
      when lower(label) like '%' || lower('zmax') || '%' then 1
      when lower(label) like '%' || lower('zosyn') || '%' then 1
      when lower(label) like '%' || lower('zyvox') || '%' then 1
    else 0
    end as antibiotic
  from d_items di
  where linksto in ('inputevents_mv','inputevents_cv')
)
"""


# metavision query
query_abx_iv_mv = """
, mv as
(
  select icustay_id
  , label as antibiotic_name
  , starttime as antibiotic_time
  , ROW_NUMBER() over (partition by icustay_id order by starttime, endtime) as rn
  from inputevents_mv mv
  inner join di
      on mv.itemid = di.itemid
      and di.antibiotic = 1
  where statusdescription != 'Rewritten'
)
"""

query_abx_iv_cv = """
, cv as
(
  select icustay_id
  , label as antibiotic_name
  , charttime as antibiotic_time
  , ROW_NUMBER() over (partition by icustay_id order by charttime) as rn
  from inputevents_cv cv
  inner join di
      on cv.itemid = di.itemid
      and di.antibiotic = 1
)
"""

# join these two together
query_abtbl = """
, ab_tbl as
(
select
      ie.icustay_id, ie.hadm_id, ie.intime
    , case when mv.antibiotic_time is not null and cv.antibiotic_time is not null
        then case when mv.antibiotic_time < cv.antibiotic_time 
                then mv.antibiotic_name
            else cv.antibiotic_name
            end
        else coalesce(mv.antibiotic_name, cv.antibiotic_name)
     end antibiotic_name
    , case when mv.antibiotic_time is not null and cv.antibiotic_time is not null
        then case when mv.antibiotic_time < cv.antibiotic_time 
                then mv.antibiotic_time
            else cv.antibiotic_time
            end
        else coalesce(mv.antibiotic_time, cv.antibiotic_time)
     end antibiotic_time
from icustays ie
left join mv
    on ie.icustay_id = mv.icustay_id
left join cv
    on ie.icustay_id = cv.icustay_id
)
"""

query_antibiotics = query_ab_ditems + query_abx_iv_mv + query_abx_iv_cv + query_abtbl
