"""Build traceable CEST table transcription and source-pinned project decisions.

One-time authoring utility, never a runtime PDF/clinical inference engine.
Transcription checked against the supplied CEST.pdf page images; human_reviewed=False.
The generated package carries the full source notice below. Local IDs are not official codes.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/phase3/sector1/terminology'
PDF_SHA = '8e4157bea0c1cfa299458df7881a7fddd0713f18e2994c502f74621c454b029a'
CITATION = 'Capsule Endoscopy Structured Terminology (CEST), Endoscopy 2005; 37: 951-959 (February 16, 2002 © Given Imaging Ltd.). doi:10.1055/s-2005-870329'
NOTICE = '''Copyright and Terms of Use
Copyright © 2005 Given Imaging Ltd. (“Given Imaging”). All rights reserved.
This Capsule Endoscopy Structured Terminology (CEST) is a copyrighted work, protected by United States and worldwide copyright laws and treaties. Users of the CEST are hereby granted a free license to use, copy or distribute the CEST in connection with supporting a structured terminology for capsule endoscopy, subject to the following conditions:
1. This copyright and terms of use notice must be displayed on each and every copy of the CEST, on all manuals and other materials used in connection with the CEST, including without limitation electronic media (i. e., disks, CD-ROMs, etc.), and in all text files loaded onto electronic media or the Internet. Such copyright and terms of use notice shall not be removed, obscured, modified or otherwise hidden. All references to the CEST, in whole or in part, shall properly cite the CEST as the source for such reference. Proper citation shall read “Capsule Endoscopy Structured Terminology (CEST), Endoscopy 2005; 37: 951-959 (February 16, 2002 © Given Imaging Ltd.)”.
2. The CEST shall not be used, in whole or in part, to harm or degrade the reputation of Given Imaging or any participant in the Given Capsule Standard Working Group (the “GCSWG”).
3. Neither Given Imaging nor any participant in the GCSWG shall be liable for any errors or omissions contained in the CEST, or for any use you or any third party may make of the CEST. Given Imaging reserves the right to make changes to the CEST without notice. The information contained in the CEST is provided on an “as is” basis without any warranties, express or implied, including but not limited to the results or effects obtained through use of information, or that it is fit for any use intended or can be used without infringing the copyright or other proprietary rights of any third party. All express and implied warranties, including those relating to title, non-infringement, merchantability or fitness for a particular purpose, are hereby expressly disclaimed and excluded with respect to any information provided hereunder.'''


def encoded(value, pretty=False):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       indent=2 if pretty else None,
                       separators=(',', ': ') if pretty else (',', ':')) + ('\n' if pretty else '')).encode()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value, True))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build_package():
    terms, attributes, contexts, unbound = [], [], [], []
    ids = {}
    def term(table, page, column, section, name, kind, organ='', parent=None):
        slug = re.sub(r'[^a-z0-9]+', '_', name.casefold()).strip('_')
        tid = f'CEST2005:T{table:02d}:{slug}'
        locator = f'PDF p.{page}; printed p.{950+page}; Table {table}; {column} column; {section}; {name}'
        record = dict(term_id=tid, preferred_term=name, term_type=kind, section=section,
                      organ=organ, source_locator=locator)
        terms.append(record)
        contexts.append(dict(record, table=table, pdf_page=page, printed_page=950+page,
                             column=column, parent_term_id=parent, identifier_scope='project_local_source_locator'))
        ids[(table,name)] = tid
        return tid
    def attr(tid, name, values, kind='enumerated', location=None):
        loc = next(t['source_locator'] for t in terms if t['term_id']==tid)
        for value in values:
            attributes.append(dict(term_id=tid, attribute=name, value=value, value_kind=kind,
                                   source_locator=(location or loc) + f'; attribute {name}'))
    def simple(table,page,col,section,names,kind,organ=''):
        return [term(table,page,col,section,n,kind,organ) for n in names]
    simple(1,3,'right','Report structure',[
        'Patient name','Date of birth','Sex','Patient ID','Study date (date of procedure)',
        'Study type (capsule type)','Capsule ID','Physician/provider','Patient history',
        'Clinical indication','ICD indication','Extent of examination','Characteristics of examination',
        'Complication','Findings','Diagnostic impression','Diagnosis ICD','Recommendation'], 'report_field')
    for n,s in [('Ingestion','Ti'),('Entry into the stomach','Tg'),('Entry into the duodenum','Td'),('Entry into the colon','Tc')]:
        t=term(2,4,'right','Temporal location',n,'temporal_landmark'); attr(t,'Symbol',[s])
    for organ,site in [('Esophagus','Z line'),('Stomach','Pylorus'),('Duodenum','Papilla'),
                       ('Small bowel',None),('Distal Ileum','Ileocecal valve'),('Colon',None)]:
        p=term(3,4,'right','Organ location',organ,'anatomical_site',organ)
        if site: term(3,4,'right','Specific site',site,'anatomical_site',organ,p)
    for n,vals in [('Right upper quadrant',[]),('Left upper quadrant',[]),('Mid-vertical line',['Upper','Lower']),
                   ('Periumbilical',[]),('Mid-horizontal line',['Right','Left']),('Right lower quadrant',[]),('Left lower quadrant',[])]:
        t=term(4,4,'right','External mapping locator',n,'external_location')
        if vals: attr(t,'Attribute',vals)
    t=term(5,5,'right','Examination characteristics','Preparation','examination_characteristic')
    attr(t,'Technique',['Specify'],'free_text_placeholder')
    attr(t,'Quality',['Excellent','Adequate','Inadequate: examination completed','Inadequate: precluding a complete examination'])
    t=term(5,5,'right','Examination characteristics','Visualization','examination_characteristic')
    attr(t,'Quality',['Excellent','Inadequate Illumination','Inadequate preparation'])
    t=term(5,5,'right','Examination characteristics','Equipment malfunction','examination_characteristic')
    attr(t,'Type',['Specify'],'free_text_placeholder');attr(t,'Type',['Recorder','Capsule'])
    t=term(5,5,'right','Examination characteristics','Extent of examination','examination_characteristic')
    attr(t,'Furthest visualized location',['Use anatomical sites as proposed in Table 3'],'source_table_reference')
    # Table 6 starts on p.6 RIGHT, continues p.6 LEFT, p.7 LEFT, then p.7 RIGHT.
    dist=['Localized','Patchy','Diffuse']; extent=['Short segment','Long segment','Whole organ']; yn=['Yes','No']
    def f(name,section,page=6,col='right',defs=None):
        t=term(6,page,col,section,name,'finding','Small bowel')
        for a,v in (defs or {}).items(): attr(t,a,v)
        return t
    f('Normal','Normal')
    f('Stenosis','Lumen',defs={'Type':['Extrinsic','Benign intrinsic','Malignant intrinsic'],'Traversed':yn})
    f('Dilated','Lumen',defs={'Longitudinal extent':extent,'Wall contractions':['Present','Absent']})
    t=f('Evidence of previous surgery','Lumen',defs={'Suture material':yn});attr(t,'Type',['Specify'],'free_text_placeholder')
    f('Blood','Content',defs={'Type':['Red','Clot','Hematin']})
    f('Bile','Content')
    t=f('Parasites','Content');attr(t,'Type',['Specify'],'free_text_placeholder')
    t=f('Foreign Body','Content');attr(t,'Type',[''],'unspecified_in_source')
    t=f('Food','Content');attr(t,'Type',['Specify'],'free_text_placeholder')
    f('Feces','Content')
    for n in ['Erythematous','Pale','Edematous (congested)','Granular','Nodular']:
        f(n,'Mucosa',defs={'Distribution pattern':dist,'Longitudinal extent':extent})
    f('Atrophic','Mucosa',col='left',defs={'Distribution pattern':dist,'Longitudinal extent':extent})
    f('Abnormal villi','Mucosa',col='left',defs={'Shape':['Convoluted','Swollen','Blunted','Absent'],'Color':['Whitish','Yellow'],
                                             'Distribution pattern':dist,'Longitudinal extent':extent})
    f('Spot','Flat lesions',col='left',defs={'Number':['Single','Multiple'],'Type':['Red','White','Black'],'Bleeding':yn,
                                           'Distribution pattern':dist,'Longitudinal extent':extent})
    f('Plaque','Flat lesions',col='left',defs={'Number':['Single','Multiple'],'Type':['Red','White','Black'],
                                             'Distribution pattern':dist,'Longitudinal extent':extent})
    t=f('Angiectasia','Flat lesions',col='left',defs={'Number':['Single','Multiple'],'Size':['Small','Medium','Large'],
        'Arborization':yn,'Bleeding':yn,'Stigmata of bleeding':yn,'Bleeding potential':['Yes','Possible','No']})
    cont='PDF p.7; printed p.957; Table 6; left column; continuation of Angiectasia from p.956'
    attr(t,'Distribution pattern',dist,location=cont);attr(t,'Longitudinal extent',extent,location=cont)
    f('Nodule','Protruding lesions',7,'left',{'Number':['Single','Few','Multiple'],'Bleeding':yn,'Stigmata of bleeding':yn,
                                          'Distribution pattern':dist,'Longitudinal extent':extent})
    f('Polyp(s)','Protruding lesions',7,'left',{'Number':['Single','Few','Multiple'],'Size':['Small','Medium','Large'],
                                            'Pedicle':['Sessile','Pedunculated','Unknown'],'Bleeding':yn})
    f('Mass/tumor','Protruding lesions',7,'left',{'Size':['Small','Medium','Large'],'Type':['Submucosal','Fungating','Ulcerated','Frond-like/villous'],
                                              'Bleeding':yn,'Stigmata of bleeding':yn})
    t=f('Venous structure','Protruding lesions',7,'left',{'Type':['Venous lake','Bleb','Varix'],'Number':['Single','Few','Multiple'],
                                                       'Bleeding':yn,'Stigmata of bleeding':yn})
    cont='PDF p.7; printed p.957; Table 6; right column; continuation of Venous structure from left column'
    attr(t,'Bleeding potential',['Yes','Possible','No'],location=cont)
    attr(t,'Distribution pattern',dist,location=cont);attr(t,'Longitudinal extent',extent,location=cont)
    f('Aphtha','Excavated lesion',7,'right',{'Number':['Single','Few','Multiple'],'Distribution pattern':dist,'Longitudinal extent':extent})
    for n in ['Erosion','Ulcer']:
        f(n,'Excavated lesion',7,'right',{'Number':['Single','Few','Multiple'],'Bleeding':yn,'Stigmata of bleeding':yn,
                                       'Distribution pattern':dist,'Longitudinal extent':extent})
    f('Scar','Excavated lesion',7,'right')
    t=f('Diverticulum','Excavated lesion',7,'right')
    unbound.append(dict(term_id=t,source_locator='PDF p.7; printed p.957; Table 6; right column; Diverticulum',
                        values=['Single','Multiple'],reason='Attribute heading blank in source; do not invent Number'))
    t=term(7,8,'left','Complications','Capsule retention','complication')
    attr(t,'Site of retention',['Use anatomical terms proposed in Table 3'],'source_table_reference')
    attr(t,'Cause of retention',['Gastroparesis','Gut stenosis','Other','Unknown'])
    attr(t,'Treatment undertaken',['None','Medical','Endoscopic removal','Elective surgery','Emergency surgery'])
    t=term(7,8,'left','Complications','Aspiration','complication');attr(t,'Treatment undertaken',['None','Endoscopic removal','Emergency surgery'])
    simple(8,8,'right','Symptoms',['Abdominal pain','Chronic diarrhea','Weight loss','Use of NSAIDs','Melena'],'indication')
    simple(8,8,'right','Signs',['Anemia'],'indication')
    t=term(8,8,'right','Signs','Abnormal imaging','indication')
    unbound.append(dict(term_id=t,source_locator='PDF p.8; printed p.958; Table 8; Abnormal imaging',values=['Specify'],
                        reason='Attribute heading blank in source; retain free-text instruction without inventing a clinical attribute'))
    t=term(8,8,'right','Signs','Obscure digestive bleeding','indication');attr(t,'Type',['Overt','Occult'])
    certainty=['Suspicion of','Established','Exclusion of','Follow-up of']
    for n in ['Celiac disease',"Crohn’s disease",'Polyposis','Intestinal tumor/mass','Intestinal ischemia']:
        t=term(9,8,'right','Diseases / reasons for examination',n,'indication')
        attr(t,'Aim',certainty,location=f'PDF p.8; printed p.958; Table 9; shared Aim column for disease list; {n}')
    for n in ['Normal','Erosion','Ulcer','Angiectasia','Tumor','Bleeding of unknown origin','Celiac disease',"Crohn’s disease",'NSAID enteritis']:
        t=term(10,9,'left','Main diagnoses',n,'diagnosis')
        attr(t,'Level of certainty',certainty,location=f'PDF p.9; printed p.959; Table 10; shared certainty column; {n}')
    for n in ['Benign','Malignant']:term(10,9,'left','Main diagnoses / Tumor',n,'diagnosis_qualifier',parent=ids[(10,'Tumor')])
    others=["Brunner’s gland hyperplasia","Dieulafoy’s lesion",'Diverticulum','Enteropathy','Familial adenomatous polyposis','GIST',
        'Graft-versus-host disease','Ischemic enteritis','Hemobilia','Intestinal lymphangiectasia','Juvenile polyposis',"Kaposi’s sarcoma",'Lipoma',
        'Lymphoma','Melanoma','Neuroendocrine tumor','Parasite','Peutz-Jeghers syndrome','Phlebectasia','Polyp',
        'Post-transplant lymphoproliferative disorder','Radiation enteritis','Tropical sprue','Varices','Vasculitis','Xanthelasma']
    for n in others:
        t=term(11,9,'left','Other diagnoses',n,'diagnosis')
        attr(t,'Level of certainty',certainty,location=f'PDF p.9; printed p.959; Table 11; shared certainty column; {n}')
    for n in ['Erosive','Erythematous','Congestive','Hemorrhagic']:
        term(11,9,'left','Other diagnoses / Enteropathy',n,'diagnosis_qualifier',parent=ids[(11,'Enteropathy')])
    package={'schema_version':'mmvqa-cest-transcription-v1', 'metadata':{
        'version':'2005','citation':CITATION,'source_url':'https://doi.org/10.1055/s-2005-870329',
        'source_file':'CEST.pdf','source_sha256':PDF_SHA,'source_bytes':209520,'page_count':9,
        'copyright_notice':NOTICE,'transcription_reviewed_by':'AI-assisted source-image transcription review',
        'transcription_reviewed_at':'2026-11-16','human_reviewed':False,'clinical_validation_performed':False,
        'identifier_scope':'project_local_source_locators_not_official_CEST_codes',
        'transcription_scope':'Tables 1-11; source table terms, explicit attributes and values; prose retained as scoped notes, not an exhaustive article transcription',
        'normalization':'PDF line-wrap hyphens/spacing normalized; footnote markers separated; no clinical synonyms inferred',
        'table_coverage':list(range(1,12))},'terms':terms,'attributes':attributes,'synonyms':[],
        'source_contexts':contexts,'unbound_source_values':unbound,
        'source_notes':[
          {'source_locator':'p.955, section 4.1','note':'Finding vocabulary is scoped to small bowel; esophageal, gastric and colonic findings refer to MST.'},
          {'source_locator':'p.955, Table 5 footnote','note':'Quality uses three equal small-bowel transit-time segments; this is not a measured image-level label.'},
          {'source_locator':'p.957, Table 6 footnote *','note':'Bleeding potential is an interpreter probability assessment, not observed bleeding.'},
          {'source_locator':'p.957, Table 6 footnote **','note':'Nodule is described as generally round, 2-3 mm, without a clearly demarcated base; distinct from nodular mucosa. This definition supplies no measurement of a dataset observation.'},
          {'source_locator':'p.959, Tables 10-11','note':'Certainty values are retained verbatim, including Exclusion of and Follow-up of; no status is asserted for a dataset image.'}],
        'mapping_approval_included':False}
    return package,ids


def build(root=ROOT):
    out=Path(root)/'outputs/phase3/sector1/terminology'
    package,ids=build_package(); path=out/'cest_2005_package.json';write(path,package)
    digest=sha(path.read_bytes())
    fingerprint=sha(b'mmvqa-cest-source-snapshot-v1\0'+bytes.fromhex(PDF_SHA)+bytes.fromhex(digest))
    manifest={'schema_version':'mmvqa-cest-github-manifest-v1','artifact':{
        'path':'outputs/phase3/sector1/terminology/cest_2005_package.json','bytes':path.stat().st_size,'sha256':digest},
        'package_content_sha256':sha(encoded(package)),'original_document_sha256':PDF_SHA,
        'original_document_bytes':209520,'source_fingerprint':fingerprint,'copyright_notice':NOTICE,'citation':CITATION}
    write(out/'cest_2005_manifest.json',manifest)
    rag_path=out/'phase3_01_rag_evidence.json'; rag=json.loads(rag_path.read_text())
    if any(s['source_id']=='CEST_2005_ORIGINAL' for s in rag['sources']):
        raise ValueError('CEST assets already integrated; do not rewrite a reviewed policy silently')
    rag['sources'].append({'source_id':'CEST_2005_ORIGINAL','title':'Korman et al. Capsule Endoscopy Structured Terminology (CEST)',
        'url':'https://doi.org/10.1055/s-2005-870329','source_role':'terminology_source',
        'independent_of_dataset':True,'year':2005,'copyright_notice':NOTICE})
    rag['passages'].append({'passage_id':'passage:CEST_2005_TABLES','source_id':'CEST_2005_ORIGINAL',
        'location':'Tables 3, 5, 6, 10 and 11, pp.954-959',
        'paraphrase':'CEST separates anatomical sites, examination characteristics, observed small-bowel findings and diagnostic impressions. Table 6 describes Blood with Type Red/Clot/Hematin, Erythematous mucosa, Foreign Body, Polyp(s), Erosion and Ulcer. Tables 10-11 contain separate diagnosis-role entries. Table 3 includes Pylorus, Papilla and Ileocecal valve. Source categories and attributes do not establish observations in dataset images.'})
    terms={t['term_id']:t for t in package['terms']}; profiles={p['class_label']:p for p in rag['profiles']}
    # Scoped project decisions, not equivalence claims made by the CEST authors.
    decisions=[
        ('ampulla of vater',3,'Papilla','related',[], 'Papilla is an endoscopic landmark related to the source class, not an exact anatomical synonym of the ampulla.'),
        ('pylorus',3,'Pylorus','exact',[], 'Same generic anatomical landmark; no additional anatomical qualifier is introduced.'),
        ('ileocecal valve',3,'Ileocecal valve','exact',[], 'Same generic anatomical landmark; source annotation provenance remains mandatory.'),
        ('angiectasia',6,'Angiectasia','exact',[], 'Same observed lesion type. Bleeding and bleeding potential remain separate unobserved schema attributes.'),
        ('erosion',6,'Erosion','exact',[], 'Same generic observed lesion; no extent, count or bleeding value is inferred.'),
        ('ulcer',6,'Ulcer','exact',[], 'Same generic observed lesion in the findings table, distinct from a complete diagnostic impression.'),
        ('polyp',6,'Polyp(s)','exact',[], 'The singular/plural source notation denotes the generic polyp finding; no number, size or histology is added.'),
        ('foreign body',6,'Foreign Body','exact',[], 'Same observed luminal-content class; no capsule retention or disorder is inferred.'),
        ('erythema',6,'Erythematous','related',[], 'Retain the mucosal descriptor as contextual vocabulary, consistent with the existing adjective policy; not an unrestricted interchangeable identifier.'),
        ('lymphangiectasia',11,'Intestinal lymphangiectasia','related',[], 'Diagnosis-role context only; do not substitute a diagnostic conclusion for an annotated visual finding.'),
        ('blood hematin',6,'Blood','compositional',[{'attribute':'Type','value':'Hematin'}], 'Preserve the annotated Hematin qualifier using the explicitly listed CEST Type value. Not an interchangeable unqualified Blood label.'),
        ('blood fresh',6,'Blood','related',[], 'Contextual Blood association only. CEST Red is a color descriptor, not automatically equivalent to the source Fresh qualifier; neither drop the source qualifier nor infer Type=Red.'),
    ]
    for label,table,name,rel,constraints,reason in decisions:
        target=terms[ids[(table,name)]]
        rag['target_decisions'].append({'class_label':label,'terminology':'CEST','target_id':target['term_id'],
            'target_term':target['preferred_term'],'target_type':target['term_type'],'source_sha256':fingerprint,
            'source_locator':target['source_locator'],'source_url':package['metadata']['source_url'],
            'decision':'approved','relation':rel,'required_attribute_constraints':constraints,
            'rationale':reason,'passage_ids':sorted(set(profiles[label]['passage_ids']+['passage:CEST_2005_TABLES'])),
            'review_method':'AI_source_grounded_project_policy','human_reviewed':False,'decision_is_inference':True})
    # Diagnosis duplicates are retained as explicit contextual associations, not aliases.
    for label,table,name in [('angiectasia',10,'Angiectasia'),('erosion',10,'Erosion'),('ulcer',10,'Ulcer'),('polyp',11,'Polyp')]:
        target=terms[ids[(table,name)]]
        rag['target_decisions'].append({'class_label':label,'terminology':'CEST','target_id':target['term_id'],
            'target_term':target['preferred_term'],'target_type':'diagnosis','source_sha256':fingerprint,
            'source_locator':target['source_locator'],'source_url':package['metadata']['source_url'],
            'decision':'approved','relation':'related','required_attribute_constraints':[],
            'rationale':'Separate diagnosis-role terminology context only; no certainty, etiology or patient diagnosis is asserted.',
            'passage_ids':sorted(set(profiles[label]['passage_ids']+['passage:CEST_2005_TABLES'])),
            'review_method':'AI_source_grounded_project_policy','human_reviewed':False,'decision_is_inference':True})
    rag['cest_source_policy']={'version':'cest-source-pinned-v1','source_fingerprint':fingerprint,
        'package_sha256':digest,'original_document_sha256':PDF_SHA,'copyright_notice':NOTICE,
        'native_only_classes_preserved':['normal clean mucosa','reduced mucosal view']}
    rag['review_date']='2026-11-16'
    rag['evidence_version']='2026-11-16.cest1'
    rag['inputs'].append({'file':'outputs/phase3/sector1/terminology/cest_2005_package.json','sha256':digest,'role':'source_checked_transcription'})
    rag['limitations']=[('No official CEST/SNOMED concept codes are invented. CEST2005 IDs are project-local source locators, not official codes; the imported source and runtime checks remain required.' if item.startswith('No CEST/SNOMED concept IDs') else item) for item in rag['limitations']]
    rag['summary'].update(sources=len(rag['sources']),passages=len(rag['passages']),target_decisions=len(rag['target_decisions']),
        approved_targets=sum(d['decision']=='approved' for d in rag['target_decisions']),
        rejected_targets=sum(d['decision']=='rejected' for d in rag['target_decisions']),cest_source_pinned_decisions=16)
    write(rag_path,rag)
    mpath=out/'rag_evidence_manifest.json';m=json.loads(mpath.read_text())
    m['artifact'].update(bytes=rag_path.stat().st_size,sha256=sha(rag_path.read_bytes()))
    m.update(bundle_content_sha256=sha(encoded(rag)),bundle_review_date=rag['review_date']);write(mpath,m)
    fixture=out/'ulcer_policy_regression.json';f=json.loads(fixture.read_text())
    f['policy_reference']['bundle_content_sha256']=m['bundle_content_sha256'];write(fixture,f)
    (out/'CEST_COPYRIGHT.txt').write_text(NOTICE+'\n',encoding='utf-8')
    print(f'CEST: {len(package["terms"])} terms, {len(package["attributes"])} attribute values, 16 project mapping decisions')
    print('source_fingerprint',fingerprint)

if __name__=='__main__': build()
