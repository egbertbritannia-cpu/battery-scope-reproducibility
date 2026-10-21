"""One-host Kaggle CPU budget cohort. No local result import or outer labels."""
from __future__ import annotations
import gc,hashlib,importlib.util,json,os,sys,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'budget_cpu_kaggle_results'
RELEASE=ROOT/'delivery'
ARCHIVE=Path('/kaggle/working/batteryscope_shared_input_budget_cpu_v1_results.zip')

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,msg):
    if not ok:raise RuntimeError('KAGGLE_CPU_CURVE:'+msg)
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(q,p)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def checkpoint_archive():
    if not OUT.exists():return
    tmp=ARCHIVE.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.partial':z.write(p,'batteryscope_shared_input_budget_cpu_v1/'+p.relative_to(OUT).as_posix())
    os.replace(tmp,ARCHIVE)

def main():
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    require(not OUT.exists(),'FRESH_OUTPUT_REQUIRED');OUT.mkdir()
    try:
        p=read(RELEASE/'protocol.frozen.json');require(sha(RELEASE/'protocol.frozen.json')==read(RELEASE/'protocol_seal.json')['sha256'],'PROTOCOL_SEAL')
        for path,b in p['embedded_files'].items():require(sha(ROOT/path)==b['sha256'],'CODE_OR_METADATA_DRIFT:'+path)
        import numpy as np,pandas as pd,pyarrow,sklearn,torch,psutil,platform
        observed=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__,
            scikit_learn=sklearn.__version__,torch=torch.__version__)
        write(OUT/'environment.json',dict(observed=observed,expected=p['expected_kaggle_cpu_environment'],platform=platform.platform(),
            CPU_count=psutil.cpu_count(),CPU_affinity=psutil.Process().cpu_affinity() if hasattr(psutil.Process(),'cpu_affinity') else None,
            CPU_clock='time.process_time',CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu_used=False))
        require(observed==p['expected_kaggle_cpu_environment'],'NUMERICAL_ENVIRONMENT_MISMATCH_NO_INSTALL')
        torch.set_num_threads(2);torch.set_num_interop_threads(1);pyarrow.set_cpu_count(2);pyarrow.set_io_thread_count(2)
        bootstrap=load('budget_cpu_mount',ROOT/'kaggle_bootstrap.py');bundle=bootstrap.resolve_bundle('/kaggle/input',ROOT)
        for name,b in p['dataset_inputs'].items():require(sha(bundle/name)==b['sha256'],'DATASET_HASH:'+name)
        old=load('budget_cpu_reader',ROOT/'protocol_and_provenance/neural_classical_fairness_v1/run_shared_input_diagnostic.py')
        curve=load('budget_cpu_curve',ROOT/'protocol_and_provenance/neural_classical_fairness_v1/run_shared_input_budget_curves_v3.py')
        old.BUNDLE=bundle;old.OUT=OUT;old.write=write
        curve.OUT=OUT;curve.PARENT=ROOT/'parent_matrices'
        curve.write=write
        require(curve.RUNG_SECONDS==p['budget_curve']['cpu_seconds'],'RUNG_DRIFT')
        class RemoteRun(old.Run):
            def __init__(self):
                self.p=p;self.started=time.monotonic();self.access=[];self.phase='setup'
                f=pd.read_parquet(bundle/'phase2/battery_phase2_fold_membership.parquet')
                f=f[f.evaluation_mode.eq('LODO')&f.outer_entity.eq('CALB')].copy()
                require(f.canonical_cell_id.is_unique,'MEMBERSHIP_DUPLICATES')
                require((f.groupby('duplicate_family_id').split.nunique()==1).all(),'OUTER_FAMILY_ROLE_OVERLAP')
                self.members=f;self.dev=f[f.split.eq('development')];require(not self.dev.source.eq('CALB').any(),'CALB_IN_DEV')
                self.roles={};config=read(bundle/'inner_config.frozen.json')
                for fold,block in enumerate(p['source_blocks']):
                    val=self.dev[self.dev.source.isin(block)];train=self.dev[~self.dev.source.isin(block)]
                    train=train[~train.duplicate_family_id.isin(val.duplicate_family_id)]
                    tr=set(train.canonical_cell_id);va=set(val.canonical_cell_id)
                    expected=next(r for r in config['inner_contexts'] if r['source']=='CALB' and r['inner_fold']==fold)
                    require(old.digest_ids(tr)==expected['train_cell_hash'] and old.digest_ids(va)==expected['validation_cell_hash'],'INNER_ROLE_HASH')
                    require(not tr&va and not set(train.duplicate_family_id)&set(val.duplicate_family_id),'INNER_FAMILY_OVERLAP')
                    self.roles[fold]=dict(train=tr,validation=va)
                write(OUT/'roles.json',{str(k):{r:sorted(v) for r,v in d.items()} for k,d in self.roles.items()})
            def check_protected(self):
                # The old scoring workspace is not mounted or written on Kaggle.
                # Protect mounted inputs and delivered code/metadata instead.
                for name,b in p['dataset_inputs'].items():require(sha(bundle/name)==b['sha256'],'MOUNTED_INPUT_CHANGED')
                for path,b in p['embedded_files'].items():require(sha(ROOT/path)==b['sha256'],'DELIVERY_CHANGED')
            def guard(self):
                begin=time.monotonic()
                while psutil.virtual_memory().available<.75*2**30:
                    require(not (OUT/'STOP').exists(),'AUTHOR_STOP')
                    require(time.monotonic()-self.started<10800,'TOTAL_TIME_CAP')
                    require(time.monotonic()-begin<900,'LOW_MEMORY_WAIT_LIMIT')
                    write(OUT/'resource_pause.json',dict(state='WAITING_AVAILABLE_RAM',at_utc=curve.now(),phase=self.phase))
                    time.sleep(5)
                super().guard()
        write(OUT/'protocol.frozen.json',p);write(OUT/'protocol_seal.json',read(RELEASE/'protocol_seal.json'))
        run=RemoteRun();write(OUT/'started.json',dict(at_utc=curve.now(),pid=os.getpid(),cohort='ALL_144_KAGGLE_CPU_SAME_HOST',
            imported_local_jobs=0,local_30_preserved_separately=True))
        checkpoint_archive();done=[]
        for fold in [0,1,2]:
            for target in ['soh30','rul90']:
                run.phase=f'data_fold{fold}_{target}';run.guard();data,state=curve.full_matrix(old,run,fold,target)
                for seed in [2026,2027,2028]:
                    for i in range(4):
                        for family in (['MLP','Extra Trees'] if seed%2==0 else ['Extra Trees','MLP']):
                            cfg=p['stage_b']['mlp_grid' if family=='MLP' else 'et_grid'][i]
                            ident=f'fold{fold}_{target}_seed{seed}_{cfg["id"]}';run.phase=ident;run.guard()
                            path=OUT/'jobs'/f'{ident}.json';require(not path.exists(),'NO_IMPLICIT_MODEL_RETRY')
                            result=curve.fit_curve(old,run,data,state,family,cfg,seed,ident)
                            result.update(id=ident,fold=fold,target=target,seed=seed,family=family,candidate=cfg,
                                train_sample_hash=state['train_sample_hash'],validation_sample_hash=state['full_validation_sample_hash'],
                                execution_cohort='ALL_144_KAGGLE_CPU_SAME_HOST')
                            write(path,result);done.append(dict(id=ident,sha256=sha(path),status=result['status']))
                            write(OUT/'completed_jobs.json',dict(jobs=done,completed=len(done),expected=144))
                            checkpoint_archive();print(json.dumps(dict(id=ident,status=result['status'],completed=len(done),expected=144)),flush=True);gc.collect()
                del data;gc.collect()
        result=curve.report(run)
        write(OUT/'execution_scope.json',dict(local_jobs_imported=0,new_Kaggle_fits=144,execution_cohort='ALL_144_KAGGLE_CPU_SAME_HOST',
            source_code_bound=True,old_local_scoring_workspace_mounted=False,old_local_scores_written=False,
            runtime_mounted_inputs_unchanged=True,cross_host_score_pooling=False))
        print(json.dumps(dict(status=result['status'],jobs=144)),flush=True)
    except Exception:
        write(OUT/'failure.json',dict(at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),traceback=traceback.format_exc(),partial_receipts_preserved=True))
        raise
    finally:checkpoint_archive()
if __name__=='__main__':main()
