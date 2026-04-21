# Sentence-chunk output samples

Representative samples from `data/mtsamples_docs/*.json` after the full ingestion pipeline. Each sentence carries:

- **Doc-level**: `specialty_cui`, each `alt_specialties[i].specialty_cui`, `doctype_cui`
- **Section-level**: `section_cui`
- **Sentence-level**: each matched entity surface form → its UMLS `cui` + `cui_name` → its semantic `tui`s + `tui_names`

Produced by `python/ingestion/mtsamples/chunk_sentences.py` (step 13 of `docs/ingestion.md`) on top of the CUI/TUI-resolved per-doc JSONs written by `link_entities_to_cui.py`.

## Sample 1 — 2104.json: Gen Med Consult (AIDS)

```
DOC  2104.json  'Gen Med Consult - 24'
  specialty:    'General Medicine'          -> C0086343
    alt:        'Consult - History and Phy.' -> C1260030
  doctype_cui:  C0743221

  SECTION  'CHIEF COMPLAINT'                 section_cui = C0277786
    SENTENCE: 'Headache and pain in the neck and lower back.'
      surface 'headache'                         -> C0018681  'Headache'
        tui: T184 'Sign or Symptom'
      surface 'pain in the neck and lower back'  -> C2129263  'lower back pain worse in morning'
        tui: T184 'Sign or Symptom'

  SECTION  'HISTORY OF PRESENT ILLNESS'      section_cui = C0262512
    SENTENCE: 'The patient is a 34 year old white man with AIDS (CD4 -67, VL -341K)
               and Castleman's Disease who presents to the VA Hospital complaining of...'
      surface 'aids'            -> C0001175  'Acquired Immunodeficiency Syndrome'
        tui: T047 'Disease or Syndrome'
      surface 'cd4'             -> C0003323  'CD4 Antigens'
        tui: T192 'Receptor', T129 'Immunologic Factor', T116 'Amino Acid, Peptide, or Protein'
      surface 'headaches'       -> C0018681  'Headache'
        tui: T184 'Sign or Symptom'
      surface 'lower back pain' -> C0024031  'Low Back Pain'
        tui: T184 'Sign or Symptom'
      surface 'neck pain'       -> C0007859  'Neck Pain'
        tui: T184 'Sign or Symptom'
      surface 'vl'              -> C0228339  'Ventral Lateral Thalamic Nucleus'
        tui: T023 'Body Part, Organ, or Organ Component'

    SENTENCE: 'He was hospitalized 3 months prior to his current presentation with
               abdominal pain and diffuse lymphadenopathy.'
      surface 'abdominal pain'         -> C0000737  'Abdominal Pain'
        tui: T184 'Sign or Symptom'
      surface 'diffuse lymphadenopathy' -> C0497156  'Lymphadenopathy'
        tui: T047 'Disease or Syndrome'

  SECTION  'FAMILY HISTORY'                  section_cui = C0241889
    SENTENCE: 'There was no history of hypertension, coronary artery disease, stroke,
               cancer or diabetes.'
      surface 'cancer'                  -> C0006826  'Malignant Neoplasms'
        tui: T191 'Neoplastic Process'
      surface 'coronary artery disease' -> C0010054  'Coronary Arteriosclerosis'
        tui: T047 'Disease or Syndrome'
      surface 'diabetes'                -> C0011847  'Diabetes'
        tui: T047 'Disease or Syndrome'
      surface 'hypertension'            -> C0020538  'Hypertensive disease'
        tui: T047 'Disease or Syndrome'
      surface 'stroke'                  -> C0038454  'Cerebrovascular accident'
        tui: T047 'Disease or Syndrome'

  SECTION  'REVIEW OF SYSTEMS'               section_cui = C0489633
    SENTENCE: 'No chest pain, palpitations, shortness of breath or coughing.'
      surface 'chest pain'          -> C0008031  'Chest Pain'
        tui: T184 'Sign or Symptom'
      surface 'coughing'            -> C0010200  'Coughing'
        tui: T184 'Sign or Symptom'
      surface 'palpitations'        -> C0030252  'Palpitations'
        tui: T033 'Finding'
      surface 'shortness of breath' -> C0013404  'Dyspnea'
        tui: T184 'Sign or Symptom'

  SECTION  'STUDIES'                         section_cui = C0947630
    SENTENCE: 'Hila and mediastinum are not enlarged.,CT Head with and without
               contrast (12/8): Ventriculomegaly and potentially minor hydrocephalus.'
      surface 'ct head with and without contrast' -> C2029583  'Computed tomography of head with contrast'
        tui: T060 'Diagnostic Procedure'
      surface 'enlarged'                          -> C0442800  'Enlarged'
        tui: T080 'Qualitative Concept'
      surface 'potentially minor hydrocephalus'   -> C1273512  'Potentially abnormal'
        tui: T033 'Finding'
      surface 'ventriculomegaly'                  -> C1531647  'Cerebral ventriculomegaly'
        tui: T033 'Finding'
```

## Sample 2 — 2014.json: Hematology Consult

```
DOC  2014.json  'Hematology Consult - 1'
  specialty:    'Hematology - Oncology'       -> C1518927
    alt:        'Consult - History and Phy.'  -> C1260030
  doctype_cui:  ''

  SECTION  'HISTORY OF PRESENT ILLNESS'      section_cui = C0262512
    SENTENCE: 'The patient is well known to me for a history of iron-deficiency anemia
               due to chronic blood loss from colitis.'
      surface 'chronic blood loss'     -> C0333278  'Chronic hemorrhage'
        tui: T046 'Pathologic Function'
      surface 'colitis'                -> C0009319  'Colitis'
        tui: T047 'Disease or Syndrome'
      surface 'iron-deficiency anemia' -> C0162316  'Iron deficiency anemia'
        tui: T047 'Disease or Syndrome'

  SECTION  'LABORATORY DATA'                 section_cui = C0677148
    SENTENCE: 'Labs today showed a white blood count of 7.9, hemoglobin 11.0,
               hematocrit 32.8, and platelets 1,121,000.'
      surface 'a white blood count' -> C1821144  'White blood count'
        tui: T033 'Finding'
      surface 'hematocrit'          -> C0018935  'Hematocrit Measurement'
        tui: T059 'Laboratory Procedure'
      surface 'labs'                -> C0587081  'Laboratory test finding'
        tui: T034 'Laboratory or Test Result'
      surface 'platelets'           -> C0032181  'Platelet count (procedure)'
        tui: T059 'Laboratory Procedure'

    SENTENCE: 'MCV is 89.'
      surface 'mcv' -> C0524587  'Mean Corpuscular Volume'
        tui: T034 'Laboratory or Test Result'

  SECTION  'CURRENT MEDICATIONS'             section_cui = C3261373
    SENTENCE: 'She is on heparin flushes, loperamide, niacin, pantoprazole, Diovan,
               Afrin nasal spray, caspofungin, daptomycin, Ertapenem, fentanyl or
               morphine p...'
      surface 'afrin nasal spray' -> C1235834  'oxymetazoline Nasal Spray [Afrin]'
        tui: T200 'Clinical Drug'
      surface 'caspofungin'       -> C0537894  'caspofungin'
        tui: T121 'Pharmacologic Substance', T116 'Amino Acid, Peptide, or Protein'
      surface 'daptomycin'        -> C0057144  'daptomycin'
        tui: T195 'Antibiotic', T116 'Amino Acid, Peptide, or Protein'
      surface 'diovan'            -> C0719949  'Diovan'
        tui: T121 'Pharmacologic Substance', T109 'Organic Chemical'
      surface 'ertapenem'         -> C1120106  'ertapenem'
        tui: T195 'Antibiotic', T109 'Organic Chemical'
      surface 'fentanyl'          -> C0015846  'fentanyl'
        tui: T121 'Pharmacologic Substance', T109 'Organic Chemical'
      surface 'heparin flushes'   -> C0354589  'heparin flush'
        tui: T121 'Pharmacologic Substance', T109 'Organic Chemical'
      surface 'loperamide'        -> C0023992  'loperamide'
        tui: T121 'Pharmacologic Substance', T109 'Organic Chemical'
      surface 'morphine'          -> C0026549  'morphine'
        tui: T121 'Pharmacologic Substance', T109 'Organic Chemical'
      surface 'niacin'            -> C0027996  'niacin'
        tui: T127 'Vitamin', T121 'Pharmacologic Substance', T109 'Organic Chemical'
      surface 'pantoprazole'      -> C0081876  'pantoprazole'
        tui: T121 'Pharmacologic Substance', T109 'Organic Chemical'

  SECTION  'VITAL SIGNS'                     section_cui = C0150404
    SENTENCE: 'Today, temperature is 98.5, pulse 99, respirations 16,
               blood pressure 105/65, and pulse is 95.'
      surface 'blood pressure' -> C0005823  'Blood Pressure'
        tui: T040 'Organism Function'
      surface 'pulse'          -> C0034107  'Pulse taking'
        tui: T058 'Health Care Activity'
      surface 'respirations'   -> C0035203  'Respiration'
        tui: T039 'Physiologic Function'
      surface 'temperature'    -> C0039476  'Temperature'
        tui: T081 'Quantitative Concept'
```

## Sample 3 — 1427.json: MRI Brain & Cerebral Angiogram

Shows the full abbreviation-to-concept chain: source writes `ASA`, step 7 expands to `aspirin`, step 8 NER picks it up as a TREATMENT entity, step 9 stamps the expansion, step 12 links it to `C0004057` (`aspirin`, T121 Pharmacologic Substance), step 13 surfaces it in the enclosing sentence.

```
DOC  1427.json  'MRI Brain & Cerebral Angiogram'
  specialty:    'Radiology'   -> C0034599
    alt:        'Neurology'   -> C0027855
  doctype_cui:  ''

  SECTION  'MEDS'                            section_cui = C0013227
    SENTENCE: 'Ortho-Novum 7-7-7 (started 2/3/96), and ASA (started 2/20/96).'
      surface 'asa'         -> C0004057  'aspirin'
        tui: T121 'Pharmacologic Substance', T109 'Organic Chemical'
      surface 'ortho-novum' -> C0722328  'Ortho Novum'
        tui: T121 'Pharmacologic Substance', T109 'Organic Chemical'

  SECTION  'PMH'                             section_cui = C0262926
    SENTENCE: '1)ventral hernia repair 10 years ago, 2)mild "concussion" suffered
               during a MVA; without loss of consciousness, 5/93, 3) Anxiety disorder,
               4) One c...'
      surface 'anxiety disorder'      -> C0003469  'Anxiety Disorders'
        tui: T048 'Mental or Behavioral Dysfunction'
      surface 'loss of consciousness' -> C0041657  'Unconscious State'
        tui: T033 'Finding'
      surface 'mild "concussion'      -> C0006107  'Brain Concussion'
        tui: T037 'Injury or Poisoning'
      surface 'ventral hernia repair' -> C0019334  'Repair of ventral hernia'
        tui: T061 'Therapeutic or Preventive Procedure'

  SECTION  'ESR 10, RF 20, ANA 1'            section_cui = C0850980
    SENTENCE: '40, ANCA <1:40, TSH 2.0, FT4 1.73, Anticardiolipin antibody IgM
               10.8GPL units (normal <10.9), Anticardiolipin antibody IgG 14.8GPL
               (normal <22.9), S...'
      surface 'anca'                         -> C0201530  'Antineutrophil cytoplasmic antibody measurement'
        tui: T059 'Laboratory Procedure'
      surface 'anticardiolipin antibody igg' -> C2346829  'Anticardiolipin IgG Antibody'
        tui: T116 'Amino Acid, Peptide, or Protein'
      surface 'anticardiolipin antibody igm' -> C2346830  'Anticardiolipin IgM Antibody'
        tui: T116 'Amino Acid, Peptide, or Protein'
      surface 'tsh'                          -> C0040160  'thyrotropin'
        tui: T125 'Hormone', T121 'Pharmacologic Substance', T116 'Amino Acid, Peptide, or Protein'

    SENTENCE: 'Urine beta-hCG pregnancy and drug screen were negative.'
      surface 'drug screen'             -> C0373483  'Drug screen (procedure)'
        tui: T059 'Laboratory Procedure'
      surface 'urine beta-hcg pregnancy' -> C0546577  'hcg pregnancy test'
        tui: T059 'Laboratory Procedure'
```

## Notes on observed behavior

- **Multi-TUI concepts are normal**: CD4 carries three semantic types (`T192 Receptor`, `T129 Immunologic Factor`, `T116 Protein`) because UMLS tags the same concept under multiple semantic facets. Downstream consumers can filter on whichever slot matters.
- **Known linkage imperfections surface here too**: `VL` → `Ventral Lateral Thalamic Nucleus` (should be "viral load"); `PT` → `post transfusion` (should be "prothrombin time"). Both are short-form abbreviations intentionally left out of `data/clinical_abbreviations_override.json` because their clinical canon is context-dependent. Adding them to the override would pin them.
- **MTSamples comma-pseudo-paragraphs**: Stanza's mimic tokenizer splits `...without any difficulty.,Two weeks later...` correctly on the `.,` boundary where the old regex used to bleed them together, but a few still slip through (e.g. `chemotherapy.,Approximately`).
