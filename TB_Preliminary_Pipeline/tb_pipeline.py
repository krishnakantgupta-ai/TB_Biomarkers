#!/usr/bin/env python3
"""Auditable GEO pilot: fetch -> prepare -> analyze. No synthetic research data."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, html, io, json, logging, os, platform, re
import shutil, sys, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
 precision_recall_curve, confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG=logging.getLogger('tb')
GROUPS={'TB','OD','LTBI','HC','UNKNOWN'}

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False))

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def retrieve(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size: return path
    LOG.info('Downloading %s',url)
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'TBResearchPilot/1.0'})
            with urllib.request.urlopen(req,timeout=180) as src,open(str(path)+'.part','wb') as dst:
                shutil.copyfileobj(src,dst)
            os.replace(str(path)+'.part',path)
            return path
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)
    return path

def bucket(acc): return acc[:-3]+'nnn'

def listing(url):
    with urllib.request.urlopen(url,timeout=120) as r: s=r.read().decode()
    return sorted(set(html.unescape(x) for x in re.findall(r'href="([^"?]+)"',s)))

def fetch(args):
    root=Path(args.work);raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
    entries=[]
    for acc in args.series:
        if not re.fullmatch(r'GSE\d+',acc):raise ValueError('Invalid GEO series')
        base=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket(acc)}/{acc}/matrix/'
        names=[n for n in listing(base) if n.endswith('_series_matrix.txt.gz')]
        if not names:raise ValueError(f'No series matrix files for {acc}')
        for name in names:
            path=retrieve(base+name,raw/name)
            entries.append({'series':acc,'file':name,'url':base+name,'sha256':sha256(path)})
    write_json(root/'downloads.json',entries)
    LOG.info('Downloaded %d matrix files. Next: prepare.',len(entries))

def parse_matrix(path):
    """Parse metadata first, then numeric matrix without loading full SOFT family."""
    meta=defaultdict(list);table=[];inside=False
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                inside=True;continue
            if line.startswith('!series_matrix_table_end'):break
            if inside:table.append(line);continue
            if line.startswith('!Sample_'):
                vals=next(csv.reader([line],delimiter='\t'))
                meta[vals[0]].append(vals[1:])
    if not table:raise ValueError(f'No expression table in {path}')
    x=pd.read_csv(io.StringIO(''.join(table)),sep='\t',index_col=0,na_values=['null','NULL'])
    x.index=x.index.astype(str);x.columns=x.columns.astype(str)
    x=x.apply(pd.to_numeric,errors='coerce')
    ids=meta['!Sample_geo_accession'][0]
    rows=[]
    for i,gsm in enumerate(ids):
        row={'sample_id':gsm}
        for key,arrays in meta.items():
            key=key.removeprefix('!Sample_')
            for j,arr in enumerate(arrays):
                if i>=len(arr):continue
                value=arr[i]
                if key.startswith('characteristics') and ':' in value:
                    k,v=value.split(':',1)
                    k='char_'+re.sub(r'[^a-z0-9]+','_',k.strip().lower()).strip('_')
                    row[k]=str(row.get(k,'')+' | '+v.strip()).strip(' |')
                else:row[key+('' if j==0 else '_'+str(j))]=value
        rows.append(row)
    return x,pd.DataFrame(rows)

def guess_metadata(row):
    """Conservative suggestions only; never silently approve unknown provenance."""
    d={str(k).lower():str(v) for k,v in row.items() if pd.notna(v)}
    def lookup(*keys):
        return ' | '.join(d.get(k,'') for k in keys).strip(' |')
    title=lookup('title').lower()
    disease=lookup('char_disease_state','char_disease_status','char_diagnosis','char_disease').lower()
    text=disease or title
    label='UNKNOWN'
    # Negated active TB is not a positive label.
    if re.search(r'latent|\bltbi\b',text):label='LTBI'
    elif re.search(r'other disease|\bnon[- ]?tb\b|\bpneumonia\b|\binfluenza\b|\basthma\b|\bsarcoidosis\b',text):label='OD'
    elif re.search(r'healthy|uninfected|\bcontrol\b',text):label='HC'
    elif re.search(r'active tuberculosis|active tb|pulmonary tuberculosis',text):label='TB'
    elif re.fullmatch(r'(tb|tuberculosis)( patient)?',text.strip()):label='TB'
    country=lookup('char_geographical_region','char_country','char_geographic_location')
    if not country:
        country=next((c for c in ['Malawi','South Africa','United Kingdom','UK'] if c.lower() in title),'UNKNOWN')
    hiv=lookup('char_hiv_status') or 'UNKNOWN'
    specimen=lookup('char_tissue','char_cell_type','source_name_ch1') or 'UNKNOWN'
    participant=lookup('char_subject_id','char_patient_id','char_participant_id','char_subject','char_patient')
    treatment=lookup('char_treatment_status','char_treatment','char_timepoint','char_time_point') or 'UNKNOWN'
    # Known series 37250 title contains WB_* sample identifier, NOT a verified participant ID.
    return dict(label=label,country=country,hiv_status=hiv,specimen=specimen,
                participant_id=participant,treatment_status=treatment)

def annotation(gpl,raw):
    override=raw/f'{gpl}_probe_to_gene.csv'
    if override.exists():
        a=pd.read_csv(override,dtype=str).fillna('')
        if not {'probe_id','gene'}.issubset(a.columns):raise ValueError(f'{override}: need probe_id,gene')
        return a.set_index('probe_id')['gene']
    url=f'https://ftp.ncbi.nlm.nih.gov/geo/platforms/{bucket(gpl)}/{gpl}/annot/{gpl}.annot.gz'
    p=retrieve(url,raw/f'{gpl}.annot.gz')
    with gzip.open(p,'rt',errors='replace') as f:
        lines=[l for l in f if l.strip() and not l.startswith(('!','#','^'))]
    a=pd.read_csv(io.StringIO(''.join(lines)),sep='\t',dtype=str).fillna('')
    cols={re.sub('[^a-z]','',c.lower()):c for c in a.columns}
    gene=next((cols[k] for k in ['genesymbol','symbol'] if k in cols),None)
    if gene is None:raise ValueError(f'No gene-symbol column in {gpl}; supply {override}')
    idcol='ID' if 'ID' in a.columns else a.columns[0]
    return a.drop_duplicates(idcol).set_index(idcol)[gene]

def map_genes(x,mapping):
    gene=mapping.reindex(x.index).fillna('').str.strip()
    # Reject ambiguous many-to-many assignments; do not pick first gene arbitrarily.
    valid=gene.ne('') & ~gene.str.contains(r'///|//|;|\||,|\s',regex=True) & ~gene.isin(['---','NA','null'])
    out=x.loc[valid].copy();out.index=gene.loc[valid].str.upper()
    return out.groupby(level=0,sort=True).median()

def prepare(args):
    root=Path(args.work);raw=root/'raw';prep=root/'prepared';prep.mkdir(parents=True,exist_ok=True)
    entries=json.loads((root/'downloads.json').read_text())
    frames=[];expr=[];coverage=[];seen=set();annotations={}
    for e in entries:
        path=raw/e['file']
        if sha256(path)!=e['sha256']:raise ValueError(f'Checksum changed: {path}')
        LOG.info('Parsing %s',path.name)
        x,m=parse_matrix(path)
        for gpl,part in m.groupby('platform_id'):
            if gpl not in annotations:annotations[gpl]=annotation(gpl,raw)
            z=map_genes(x.loc[:,part.sample_id],annotations[gpl])
            coverage.append(dict(series=e['series'],platform=gpl,probes=len(x),mapped_genes=len(z),samples=z.shape[1]))
            keep=[s for s in z.columns if s not in seen]
            if not keep:continue
            seen.update(keep);expr.append(z[keep]);part=part[part.sample_id.isin(keep)].copy()
            part['series']=e['series'];part['source_file']=e['file'];frames.append(part)
    m=pd.concat(frames,ignore_index=True).fillna('')
    if not expr:raise ValueError('No usable expression data')
    # Mapping/common universe is based only on platform gene availability, not outcome or expression statistics.
    common=sorted(set.intersection(*(set(z.index) for z in expr)))
    if len(common)<500:raise ValueError(f'Only {len(common)} common genes; inspect annotations')
    z=pd.concat([e.loc[common] for e in expr],axis=1)
    z.to_pickle(prep/'expression_genes.pkl')
    m.to_csv(prep/'metadata_raw.csv',index=False)
    pd.DataFrame(coverage).to_csv(prep/'mapping_coverage.csv',index=False)
    manifest=[]
    for _,row in m.iterrows():
        g=guess_metadata(row)
        manifest.append(dict(sample_id=row.sample_id,series=row.series,platform=row.platform_id,
            title=row.get('title',''),**g,include='no',reviewed='no',provenance_verified='no',
            inflammation_class='UNKNOWN',notes='Review source publication and participant/timepoint identifiers.'))
    mp=prep/'sample_manifest.csv'
    if mp.exists():
        pd.DataFrame(manifest).to_csv(prep/'sample_manifest_suggestions.csv',index=False)
        LOG.warning('Kept existing manifest; wrote suggestions separately.')
    else:pd.DataFrame(manifest).to_csv(mp,index=False)
    write_json(prep/'preparation.json',dict(common_genes=len(common),samples=z.shape[1],
        warning='Published processed matrices may already have cohort-level normalization. This pilot cannot undo it.',
        downloads=entries,annotation_sha256={p.name:sha256(p) for p in raw.glob('GPL*') if p.is_file()}))
    LOG.info('Prepared %d genes x %d samples. Review %s before analyze.',*z.shape,mp)

def curate37250(args):
    """Dataset-specific, transparent study-level curation, not a general label override."""
    p=Path(args.work)/'prepared/sample_manifest.csv'
    m=pd.read_csv(p,dtype=str).fillna('')
    mask=m.series.eq('GSE37250');z=m[mask]
    if len(z)!=537:raise ValueError('Expected 537 deposited GSE37250 samples; manual review required after source changes')
    expected={('TB','Malawi'):102,('TB','South Africa'):93,('OD','Malawi'):64,('OD','South Africa'):111,('LTBI','Malawi'):71,('LTBI','South Africa'):96}
    if z.groupby(['label','country']).size().to_dict()!=expected:raise ValueError('Labels/counts differ from audited metadata; manual review required')
    ids=z.title.str.extract(r'WB_[1-6]_(M|SA)_(\d+)$')
    if ids.isna().any().any():raise ValueError('Unexpected sample title; cannot derive study-code ID')
    person='GSE37250:'+ids[0]+':'+ids[1]
    if person.duplicated().any():raise ValueError('Repeated study-code IDs detected')
    backup=p.with_name('sample_manifest_before_curation.csv')
    if not backup.exists():shutil.copyfile(p,backup)
    m.loc[mask,'participant_id']=person.to_numpy()
    m.loc[mask,'treatment_status']='baseline_or_within_24h'
    m.loc[mask,['reviewed','provenance_verified','include']]='yes'
    m.loc[mask,'notes']='Dataset-specific source curation: GSE37250 metadata; Kaforou 2013 doi:10.1371/journal.pmed.1001538. Study IDs derived from WB_group_site_code titles; one deposited row per code. Blood before or within 24h of TB treatment. OD is broad other disease, not restricted to respiratory infections. No pulmonary-only assumption.'
    m.to_csv(p,index=False)
    LOG.info('Curated GSE37250 using deposited labels and study-level methods. Original manifest preserved.')

class TopVariance(BaseEstimator,TransformerMixin):
    def __init__(self,k=500):self.k=k
    def fit(self,X,y=None):
        v=np.var(X,axis=0);self.indices_=np.argsort(v,kind='stable')[-min(self.k,X.shape[1]):]
        return self
    def transform(self,X):return np.asarray(X)[:,self.indices_]

def numeric_matrix(x):
    a=x.to_numpy(dtype=np.float32).T
    # Sample-wise rank normalization avoids a fitted cross-cohort batch transformation.
    # Signed/background-subtracted intensities need no guessed log transform.
    for i in range(len(a)):
        good=np.isfinite(a[i]);n=good.sum()
        if n: a[i,good]=(rankdata(a[i,good],method='average')-.5)/n
        a[i,~good]=np.nan
    return a

def safe_cv(y,groups,seed,n=3):
    for k in range(n,1,-1):
        cv=list(StratifiedGroupKFold(k,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))
        if all(len(np.unique(y[t]))==2 and len(np.unique(y[v]))==2 for t,v in cv):return cv
    raise ValueError('Insufficient participant groups per class for CV')

def split_development(y,groups,seed):
    # One stratified group fold reserved for threshold selection only (~20-33%).
    folds=safe_cv(y,groups,seed,5)
    t,c=folds[0]
    assert not set(groups[t]) & set(groups[c])
    return t,c

def pipeline(model,k):
    return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('select',TopVariance(k)),('scale',StandardScaler()),('model',model)])

def threshold_at_sensitivity(y,p,target):
    fpr,tpr,thresholds=roc_curve(y,p)
    allowed=np.where((tpr>=target)&np.isfinite(thresholds))[0]
    if not len(allowed):raise ValueError('No finite threshold achieves target sensitivity')
    # Largest threshold satisfying sensitivity maximizes specificity on calibration set.
    return float(np.max(thresholds[allowed]))

def measures(y,p,threshold):
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    return dict(n=len(y),n_tb=int(sum(y)),n_non_tb=int(sum(1-y)),
        auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        auprc=float(average_precision_score(y,p)) if len(set(y))==2 else None,
        sensitivity=float(tp/(tp+fn)) if tp+fn else None,
        specificity=float(tn/(tn+fp)) if tn+fp else None,
        brier=float(brier_score_loss(y,p)),tp=int(tp),fn=int(fn),tn=int(tn),fp=int(fp))

def bootstrap(y,p,g,threshold,n,seed):
    rng=np.random.default_rng(seed);units=np.unique(g);values=defaultdict(list)
    loc={u:np.flatnonzero(g==u) for u in units}
    for _ in range(n):
        idx=np.concatenate([loc[u] for u in rng.choice(units,len(units),replace=True)])
        if len(set(y[idx]))<2:continue
        d=measures(y[idx],p[idx],threshold)
        for k in ['auroc','auprc','sensitivity','specificity','brier']:values[k].append(d[k])
    result={}
    for k,v in values.items():
        result[k+'_ci_low'],result[k+'_ci_high']=map(float,np.percentile(v,[2.5,97.5]))
    result['bootstrap_valid_replicates']=len(values.get('auroc',[]))
    return result

def plot_predictions(pred,out):
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for name,d in pred.groupby('model'):
        y=d.y_true.to_numpy();p=d.probability.to_numpy()
        if len(set(y))<2:continue
        f,t,_=roc_curve(y,p);axs[0].plot(f,t,label=f'{name} ({roc_auc_score(y,p):.3f})')
        prec,rec,_=precision_recall_curve(y,p);axs[1].plot(rec,prec,label=name)
        frac,mp=calibration_curve(y,p,n_bins=5,strategy='quantile');axs[2].plot(mp,frac,'o-',label=name)
    axs[0].plot([0,1],[0,1],'k--');axs[2].plot([0,1],[0,1],'k--')
    for ax,title,x,y in zip(axs,['Held-out ROC','Held-out precision-recall','Held-out reliability'],
         ['False positive rate','Recall','Mean predicted probability'],['Sensitivity','Precision','Observed TB fraction']):
        ax.set(title=title,xlabel=x,ylabel=y);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'heldout_curves.png',dpi=180);plt.close(fig)

def pca_plot(X,manifest,train,out):
    im=SimpleImputer(strategy='median',keep_empty_features=True).fit(X[train])
    a=im.transform(X);sel=TopVariance(500).fit(a[train]);a=sel.transform(a)
    sc=StandardScaler().fit(a[train]);a=sc.transform(a)
    pc=PCA(n_components=2,random_state=0).fit(a[train]);z=pc.transform(a)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,col in zip(axs,['label','country','series']):
        for name,part in manifest.groupby(col):
            idx=part.index.to_numpy();ax.scatter(z[idx,0],z[idx,1],s=12,alpha=.65,label=name)
        ax.set(title=col,xlabel='PC1 (training fit)',ylabel='PC2 (training fit)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'pca_audit.png',dpi=180);plt.close(fig)

def balanced_indices(meta,y,seed):
    # Bootstrap whole participant groups within country x class, never external rows.
    rng=np.random.default_rng(seed);cells=defaultdict(list)
    for i,r in meta.iterrows():cells[(str(r.country),int(y[i]))].append(i)
    if len(set(meta.country))<2:return None
    countries=sorted(set(meta.country))
    if any((c,k) not in cells for c in countries for k in [0,1]):
        LOG.warning('Balanced MLP skipped: a country lacks a class');return None
    n=max(len(v) for v in cells.values())
    return np.concatenate([rng.choice(v,n,replace=True) for v in cells.values()])

def analyze(args):
    root=Path(args.work);prep=root/'prepared';out=root/'results'
    if out.exists() and any(out.iterdir()):raise ValueError('Results directory exists: use a new --work copy to avoid repeated test tuning.')
    m=pd.read_csv(prep/'sample_manifest.csv',dtype=str).fillna('')
    required={'sample_id','label','country','series','participant_id','include','reviewed','provenance_verified','specimen','treatment_status'}
    if not required.issubset(m):raise ValueError('Manifest missing '+str(required-set(m)))
    if m.sample_id.duplicated().any():raise ValueError('Duplicate GSM IDs in manifest')
    include=m.include.str.lower().eq('yes')
    if not include.any():raise ValueError('No approved samples. Review manifest; set include/reviewed/provenance_verified=yes.')
    selected=m.loc[include].copy()
    if not selected.reviewed.str.lower().eq('yes').all() or not selected.provenance_verified.str.lower().eq('yes').all():
        raise ValueError('Included rows must have reviewed=yes and provenance_verified=yes')
    if not selected.label.isin(GROUPS-{'UNKNOWN'}).all():raise ValueError('Included samples require explicit TB/OD/LTBI/HC labels')
    if selected.participant_id.eq('').any():raise ValueError('Assign verified participant IDs to all included rows')
    # Global participant identifiers required across all series; do not prefix to hide overlap.
    if not selected.specimen.str.lower().str.contains(r'whole[ _-]?blood').all():raise ValueError('Primary analysis requires whole-blood samples')
    if not selected.treatment_status.str.lower().isin(['pretreatment','untreated','baseline_untreated','baseline_or_within_24h']).all():
        raise ValueError('Included rows must have documented baseline treatment status')
    controls=args.controls.split(',')
    if any(c not in ['OD','LTBI','HC'] for c in controls):raise ValueError('Invalid controls')
    selected=selected[selected.label.isin(['TB']+controls)].reset_index(drop=True)
    if selected.empty:raise ValueError('No samples for requested comparison')
    # Retain one baseline sample per globally identified participant to avoid repeated-measure weighting.
    duplicates=selected[selected.participant_id.duplicated(keep=False)]
    if len(duplicates):raise ValueError('Repeated participant IDs: choose one pretreatment sample per participant in manifest')
    expr=pd.read_pickle(prep/'expression_genes.pkl')
    if not set(selected.sample_id).issubset(expr.columns):raise ValueError('Manifest sample missing from matrix')
    selected['partition']='unused'
    if args.test_country:
        if not args.test_series:raise ValueError('--test-country also requires --test-series')
        test=(selected.series.eq(args.test_series)&selected.country.eq(args.test_country)).to_numpy()
        dev=(selected.series.eq(args.development_series)&~test).to_numpy()
    else:
        if args.test_series==args.development_series:raise ValueError('Same series requires --test-country')
        test=selected.series.eq(args.test_series).to_numpy();dev=selected.series.eq(args.development_series).to_numpy()
    if not test.any() or not dev.any():raise ValueError('Empty development or test set; check exact country/series labels')
    y=selected.label.eq('TB').astype(int).to_numpy();groups=selected.participant_id.to_numpy()
    for tag,idx in [('development',dev),('test',test)]:
        if min(Counter(y[idx]).values(),default=0)<8 or len(set(y[idx]))<2:raise ValueError(f'{tag}: need >=8 participants in EACH class')
    if set(groups[dev])&set(groups[test]):raise ValueError('Participant overlap across development/test')
    X=numeric_matrix(expr[selected.sample_id]);genes=np.array(expr.index)
    di=np.flatnonzero(dev);ti=np.flatnonzero(test)
    tr,ca=split_development(y[di],groups[di],args.seed);tr=di[tr];ca=di[ca]
    selected.loc[tr,'partition']='train';selected.loc[ca,'partition']='threshold_calibration';selected.loc[ti,'partition']='heldout_test'
    out.mkdir(parents=True,exist_ok=False)
    selected.to_csv(out/'sample_partitions.csv',index=False)
    m.groupby(['series','label','include'],dropna=False).size().reset_index(name='n').to_csv(out/'all_sample_counts.csv',index=False)
    selected.groupby(['partition','series','country','label','hiv_status'],dropna=False).size().reset_index(name='n').to_csv(out/'analysis_sample_counts.csv',index=False)
    pca_plot(X,selected,tr,out)
    # Exact-expression duplicate screen: training vs test on common genes.
    fingerprints={}
    for i in np.r_[tr,ca,ti]:
        fingerprint=hashlib.sha256(np.nan_to_num(X[i],nan=-1).tobytes()).hexdigest()
        if fingerprint in fingerprints:raise ValueError(f'Identical expression profiles: {selected.iloc[i].sample_id} and {fingerprints[fingerprint]}. Inspect overlap.')
        fingerprints[fingerprint]=selected.iloc[i].sample_id
    cv=safe_cv(y[tr],groups[tr],args.seed,3)
    base=pipeline(LogisticRegression(penalty='elasticnet',solver='saga',max_iter=5000,random_state=args.seed),args.genes)
    search=GridSearchCV(base,{'model__C':[.01,.1,1.0],'model__l1_ratio':[.2,.8]},cv=cv,scoring='roc_auc',n_jobs=args.jobs,error_score='raise')
    LOG.info('Tuning elastic-net on TRAIN only (%d participants)',len(tr));search.fit(X[tr],y[tr])
    pd.DataFrame(search.cv_results_).drop(columns=['params']).to_csv(out/'training_tuning_results.csv',index=False)
    models={'elastic_net':search.best_estimator_}
    if not args.skip_neural:
        # Fixed architecture and epochs: no test-driven architecture/epoch selection.
        for name,balanced in [('mlp',False),('cohort_balanced_mlp',True)]:
            fits=[]
            for seed in [args.seed,args.seed+1,args.seed+2]:
                model=pipeline(MLPClassifier(hidden_layer_sizes=(64,16),activation='relu',solver='adam',
                    alpha=.01,max_iter=args.epochs,early_stopping=False,tol=0,n_iter_no_change=args.epochs+1,
                    random_state=seed,batch_size=32,learning_rate_init=.001),args.genes)
                if balanced:
                    idx=balanced_indices(selected.iloc[tr].reset_index(drop=True),y[tr],seed)
                    if idx is None:break
                    # Fit preprocessing on original training participants, then resample transformed rows.
                    pre=Pipeline(model.steps[:-1]);a=pre.fit_transform(X[tr],y[tr])
                    model.named_steps['model'].fit(a[idx],y[tr][idx])
                else:model.fit(X[tr],y[tr])
                fits.append(model)
            if fits:models[name]=fits
    # Stability is descriptive; candidate panel is selected using TRAIN folds only.
    counts=Counter();coefsum=Counter()
    for a,b in cv:
        fit=sklearn.base.clone(search.best_estimator_)
        fit.fit(X[tr[a]],y[tr[a]])
        selected_genes=genes[fit.named_steps['select'].indices_]
        co=fit.named_steps['model'].coef_[0]
        order=np.argsort(np.abs(co))[::-1]
        chosen=[j for j in order if abs(co[j])>1e-10][:30]
        for j in chosen:counts[selected_genes[j]]+=1;coefsum[selected_genes[j]]+=abs(float(co[j]))
    stability=pd.DataFrame([dict(gene=g,selection_frequency=n/len(cv),mean_abs_coefficient=coefsum[g]/n) for g,n in counts.items()])
    if stability.empty:LOG.warning('No nonzero coefficients; compact panel skipped.')
    else:
        stability=stability.sort_values(['selection_frequency','mean_abs_coefficient'],ascending=False)
        stability.to_csv(out/'gene_stability.csv',index=False)
        panel=stability.gene.head(args.panel_genes).tolist();idx=np.array([np.where(genes==g)[0][0] for g in panel])
        # Same full common-gene rank input; this is NOT an independently deployable qPCR panel.
        compact=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),
            ('model',LogisticRegression(C=.1,solver='lbfgs',max_iter=3000,random_state=args.seed))])
        compact.fit(X[tr][:,idx],y[tr]);models['compact_candidate']=(compact,idx)
        pd.DataFrame({'gene':panel}).to_csv(out/'candidate_panel.csv',index=False)
        top=stability.head(20).iloc[::-1];fig,ax=plt.subplots(figsize=(7,6));ax.barh(top.gene,top.selection_frequency)
        ax.set(xlabel='Fraction of training folds (top 30 nonzero coefficients)',title='Exploratory feature-selection stability');fig.tight_layout();fig.savefig(out/'gene_stability.png',dpi=180);plt.close(fig)
    predictions=[];metrics=[];thresholds={}
    def predict(model,a):
        if isinstance(model,list):return np.mean([z.predict_proba(a)[:,1] for z in model],axis=0)
        if isinstance(model,tuple):return model[0].predict_proba(a[:,model[1]])[:,1]
        return model.predict_proba(a)[:,1]
    for name,model in models.items():
        pc=predict(model,X[ca]);threshold=threshold_at_sensitivity(y[ca],pc,args.target_sensitivity)
        thresholds[name]=dict(threshold=threshold,calibration=measures(y[ca],pc,threshold))
        pt=predict(model,X[ti]);row=dict(model=name,threshold=threshold,**measures(y[ti],pt,threshold))
        row.update(bootstrap(y[ti],pt,groups[ti],threshold,args.bootstrap,args.seed));metrics.append(row)
        d=selected.iloc[ti].copy();d['model']=name;d['y_true']=y[ti];d['probability']=pt;d['prediction']=(pt>=threshold).astype(int)
        predictions.append(d)
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(out/'heldout_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'heldout_metrics.csv',index=False);write_json(out/'thresholds.json',thresholds)
    plot_predictions(pred,out)
    fig,axs=plt.subplots(1,len(models),figsize=(4*len(models),4),squeeze=False)
    for ax,(name,d) in zip(axs[0],pred.groupby('model')):
        cm=confusion_matrix(d.y_true,d.prediction,labels=[0,1]);ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=14)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-TB','TB'],yticklabels=['Non-TB','TB'],xlabel='Predicted',ylabel='Observed',title=name)
    fig.tight_layout();fig.savefig(out/'confusion_matrices.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    labels=['Manifest total','Included for comparison','Training','Threshold calibration','Held-out test']
    nums=[len(m),len(selected),len(tr),len(ca),len(ti)]
    ax.barh(labels[::-1],nums[::-1]);ax.set(xlabel='Samples',title='Sample audit and analysis partitions')
    fig.tight_layout();fig.savefig(out/'sample_flow.png',dpi=180);plt.close(fig)
    subgroup=[]
    for name,d in pred.groupby('model'):
        for col in ['hiv_status','country','label','inflammation_class']:
            for value,s in d.groupby(col):
                subgroup.append(dict(model=name,subgroup=col,value=value,**measures(s.y_true.to_numpy(),s.probability.to_numpy(),thresholds[name]['threshold'])))
    pd.DataFrame(subgroup).to_csv(out/'subgroup_metrics.csv',index=False)
    joblib.dump(dict(models=models,genes=genes,thresholds=thresholds,representation='within_sample_common_gene_percentile_rank'),out/'models.joblib')
    provenance=dict(arguments=vars(args),versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        manifest_sha256=sha256(prep/'sample_manifest.csv'),expression_sha256=sha256(prep/'expression_genes.pkl'),
        warning='Exploratory public-data pilot; no clinical/qPCR validation. Published normalization can include held-out country samples. No inflammation-aware auxiliary task implemented; cohort balancing is an ablation only.')
    write_json(out/'run_provenance.json',provenance)
    comparison='TB versus '+', '.join(controls)
    lines=['# Preliminary computational results',f'Comparison: {comparison}.',
        f'Train: {len(tr)}; threshold calibration: {len(ca)}; held-out test: {len(ti)}.',
        'Thresholds were set on a separate development subset; test labels were used only for reporting.',
        'The development CV scores are tuning diagnostics, not independent validation.',
        '## Held-out results']
    for r in metrics:
        lines.append(f"- {r['model']}: AUROC {r['auroc']:.3f}; sensitivity {r['sensitivity']:.3f}; specificity {r['specificity']:.3f}; TP/FN/TN/FP {r['tp']}/{r['fn']}/{r['tn']}/{r['fp']}.")
    lines+=['## Interpretation limits','- Deposited TB and OD labels do not establish pulmonary-only TB or respiratory-only comparator disease. GSE37250 blood was collected before or within 24h of treatment.', '- Confidence intervals are participant-bootstrap intervals conditional on the fitted model and cohort; they do not quantify between-study variation.',
        '- No claim of clinical utility, causal TB-specific biology, or qPCR transfer is supported.',
        '- Published processed arrays may have been normalized together; country-held-out results are preliminary, not fully independent preprocessing.',
        '- Baseline MLP and cohort-balanced MLP are implemented; pathway-constrained or viral-inflammation auxiliary models are not.',
        '- Compact genes use ranks relative to the full common-gene universe. A new qPCR score and independent assay bridge are required.',
        '- Sensitivity target applies to calibration data only; inspect actual held-out sensitivity.',
        '- Feature stability uses only three training folds and is exploratory.',
        '- MLP probabilities have not been post-hoc probability calibrated; a reliability plot is provided.',
        '- Model selection must not be revised after viewing the held-out test. New experiments need new untouched data.']
    (out/'PRELIMINARY_REPORT.md').write_text('\n\n'.join(lines))
    LOG.info('Finished. Read %s',out/'PRELIMINARY_REPORT.md')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',default='tb_work',help='Data/output directory (put BEFORE subcommand)')
    sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fetch');f.add_argument('--series',nargs='+',default=['GSE37250']);f.set_defaults(func=fetch)
    p=sub.add_parser('prepare');p.set_defaults(func=prepare)
    c=sub.add_parser('curate37250');c.set_defaults(func=curate37250)
    a=sub.add_parser('analyze');a.add_argument('--development-series',default='GSE37250');a.add_argument('--test-series',default='GSE37250')
    a.add_argument('--test-country',default='Malawi',help='Exact manifest country. Use empty string for entire external series.')
    a.add_argument('--controls',default='OD',help='OD, LTBI or HC; comma-separated only for prespecified combined analysis')
    a.add_argument('--genes',type=int,default=500);a.add_argument('--panel-genes',type=int,default=8)
    a.add_argument('--epochs',type=int,default=200);a.add_argument('--seed',type=int,default=42)
    a.add_argument('--bootstrap',type=int,default=1000);a.add_argument('--target-sensitivity',type=float,default=.9)
    a.add_argument('--jobs',type=int,default=1);a.add_argument('--skip-neural',action='store_true');a.set_defaults(func=analyze)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    if args.command=='analyze':
        if not .5<=args.target_sensitivity<=1:parser.error('Target sensitivity must be between 0.5 and 1')
        if min(args.genes,args.panel_genes,args.epochs,args.bootstrap)<1:parser.error('Counts must be positive')
    try:args.func(args)
    except Exception as e:LOG.error('%s: %s',type(e).__name__,e);sys.exit(1)
if __name__=='__main__':
    sys.modules['tb_pipeline']=sys.modules[__name__]
    TopVariance.__module__='tb_pipeline'
    main()
