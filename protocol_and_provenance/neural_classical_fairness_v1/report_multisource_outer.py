"""Project accepted scoring aggregates into review tables and a source plot."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1 import score_multisource_outer as score
import pandas as pd
OUT=score.OUT
SYSTEMS=['TCN','GRU','Ridge','HGB','Extra Trees','Trend','Persistence','Median','Mean']
TARGETS=['soh30','rul90','rul80'];LABELS={'soh30':'SoH30','rul90':'RUL90','rul80':'RUL80'}
def fmt(x,n=4,signed=False):
 return 'NE' if pd.isna(x) else format(float(x),('+' if signed else '')+'.'+str(n)+'f').replace('.',',').replace('-','−')
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def main():
 manifest=score.read(OUT/'score_manifest.json');score.need(manifest['status']=='ALL_12_SOURCE_POINT_SCORING_COMPLETE_NO_RESELECTION','SCORING_NOT_COMPLETE')
 for name,b in manifest['outputs'].items():score.need(score.binding(OUT/name)==b,'SCORED_OUTPUT_DRIFT')
 agg=pd.read_csv(OUT/'aggregate_metrics.csv');ps=pd.read_csv(OUT/'source_summary.csv');pairs=pd.read_csv(OUT/'pairwise_summary.csv');loo=pd.read_csv(OUT/'leave_one_source_out.csv');snl=pd.read_csv(OUT/'snl_exclusion.csv');support=pd.read_csv(OUT/'support.csv');seed=pd.read_csv(OUT/'source_seed_metrics.csv')
 def aggregate(system,t):
  r=agg.loc[agg.system.eq(system)&agg.endpoint.eq(t)];return None if r.empty else r.iloc[0]
 def delta(left,right,t):
  a=ps.loc[ps.system.eq(left)&ps.endpoint.eq(t)].set_index('source').primary;b=ps.loc[ps.system.eq(right)&ps.endpoint.eq(t)].set_index('source').primary
  return (a-b).sort_index()
 def contrast(left,right,t):
  d=delta(left,right,t);tol=score.read(OUT/'protocol.frozen.json')['tolerance'];lo=[d.drop(s).mean() for s in d.index]
  return dict(left=left,right=right,endpoint=t,delta=float(d.mean()),wins=int((d < -tol).sum()),ties=int((d.abs()<=tol).sum()),losses=int((d>tol).sum()),sources=len(d),loo_min=float(min(lo)),loo_max=float(max(lo)),without_snl=float(d.drop('SNL').mean()))
 chosen=[contrast(left,right,t) for left,right in [('TCN','Trend'),('GRU','Trend'),('TCN','Extra Trees'),('GRU','Extra Trees'),('GRU','TCN')] for t in TARGETS]
 score.write(OUT/'report_contrasts.json',dict(rows=chosen,negative_favors_left=True,descriptive_only=True))
 rows=[]
 for system in SYSTEMS:
  if system=='Median':continue
  cells=[]
  for t in TARGETS:
   # Endpoint-specific naive comparator: Persistence for SoH, Median for RUL.
   name='Median' if system=='Persistence' and t!='soh30' else system;r=aggregate(name,t)
   cells.append('—' if r is None else fmt(r.primary)+((' ± '+fmt(r.primary_seed_sd)) if r.seeds>1 else ''))
  rows.append([('Persistence / Median' if system=='Persistence' else system),*cells])
 primary=table(['Hệ thống','SoH30 MAE','RUL90 RMSLE','RUL80 RMSLE'],rows)
 unitrows=[]
 for system in SYSTEMS:
  if system=='Median':continue
  for t in TARGETS:
   name='Median' if system=='Persistence' and t!='soh30' else system;r=aggregate(name,t)
   if r is not None:unitrows.append(['Persistence / Median' if system=='Persistence' else system,LABELS[t],fmt(r.mae,4 if t=='soh30' else 2),fmt(r.rmse,4 if t=='soh30' else 2)])
 units=table(['Hệ thống','Nhãn','MAE','RMSE'],unitrows)
 sourcerows=[]
 for source in sorted(support.source.unique()):
  for t in TARGETS:
   values=[]
   for system in ['TCN','GRU','Extra Trees','Trend']:
    r=ps.loc[ps.source.eq(source)&ps.endpoint.eq(t)&ps.system.eq(system)];values.append('NE' if r.empty else fmt(r.iloc[0].primary))
   sourcerows.append([source,LABELS[t],*values])
 sources=table(['Nguồn','Nhãn','TCN','GRU','Extra Trees','Trend'],sourcerows)
 sensitivity=table(['Đối chiếu','Nhãn','Chênh lệch','Thắng/hòa/thua','Bỏ một nguồn: min–max','Loại SNL'],[[r['left']+' − '+r['right'],LABELS[r['endpoint']],fmt(r['delta'],signed=True),f"{r['wins']}/{r['ties']}/{r['losses']}",fmt(r['loo_min'],signed=True)+'; '+fmt(r['loo_max'],signed=True),fmt(r['without_snl'],signed=True)] for r in chosen])
 parts=['### A.20 Đối chứng khớp lại NFR1 trên 12 nguồn','',
 'S30–S33 dùng các cấu hình đã chọn bằng fold nội bộ, không chọn lại từ kết quả ngoài. Có 651 lượt khớp ngoài và 36 job tham chiếu (108 thực thi); 687 job đã qua kiểm tra hash, miền dự báo và suy luận lại toàn bộ từ tiến trình mới cho mô hình đã học. Các phiên chuyển tiếp chỉ nhập job hoàn tất, giữ nguyên seed và cấu hình. Chúng không phải các lần lặp độc lập. Lượt chọn nội bộ dùng seed 2026; năm seed khớp ngoài có điều kiện trên lựa chọn đó. Đây là mở rộng hậu nghiệm sau khi đã xem kết quả lịch sử và chẩn đoán CALB, không phải đăng ký trước hoặc xác nhận trên dữ liệu chưa từng quan sát.','',
 '**Bảng S30. Sai số chính sau khớp lại trên 12 nguồn.** Trung bình cân đều cell trong nguồn, rồi cân đều nguồn; ± là độ lệch chuẩn mẫu giữa năm seed, không phải khoảng tin cậy. Ridge và các mốc tham chiếu tất định chỉ chạy một lần; không tạo thêm seed giả. RUL90 chỉ lấy 11 nguồn vì CALB không có nhãn chấm được. Mean là trung bình SoH hoặc trung bình trên thang log của RUL trong tập phát triển; Persistence và Median là hai mốc khác nhau theo nhãn.','',primary,'',
 '**Bảng S31. Sai số theo đơn vị gốc.** RUL tính bằng chu kỳ. RMSE được tính trong từng cell rồi lấy trung bình đều cell và nguồn, không lấy căn của MSE gộp toàn bộ mốc. Chỉ số được trung bình qua năm seed ở các hệ thống ngẫu nhiên; không gộp dự báo thành ensemble.','',units,'',
 '**Bảng S32. Sai số theo nguồn của các đối chiếu neural–mốc tham chiếu.** SoH30 dùng MAE; RUL dùng RMSLE. TCN, GRU và Extra Trees được trung bình qua năm seed; Trend có một thực thi. NE không góp vào trung bình. Số cell và mốc giữ như S24; bảng chi tiết đủ các hệ thống và seed được lưu trong đầu ra chấm.','',sources,'',
 '**Bảng S33. Đối chiếu ghép cặp và độ nhạy tổng hợp.** Âm có lợi cho vế trái. Thắng/hòa/thua đếm theo trung bình seed ở từng nguồn, với ngưỡng hòa tuyệt đối 10⁻⁸. Bỏ từng nguồn giữ nguyên mô hình và điểm từng nguồn. Loại SNL bỏ toàn bộ nguồn, rộng hơn bỏ riêng cell dùng đại diện cửa sổ SOC; đây chỉ là độ nhạy của phép tổng hợp, không khớp lại hay đổi lựa chọn nội bộ.','',sensitivity,'',
 '![Chênh lệch sai số NFR1 theo nguồn so với Extra Trees. Điểm là trung bình năm chênh lệch ghép seed; đoạn ngang là nhỏ nhất–lớn nhất giữa seed, không phải khoảng tin cậy. Âm có lợi cho mạng; SoH30 dùng MAE, RUL dùng RMSLE. CALB/RUL90 là NE.](../latex/figures/figS01_nfr1_source_deltas.png)','',
 'Đối chứng dùng cùng hợp đồng dữ liệu, phân hoạch, nhãn, miền chấm và môi trường phiên bản số; biểu diễn đầu vào, mục tiêu, lấy mẫu và tính toán vẫn khác nhau. Vì vậy, các con số so sánh các hệ thống đã khóa, không nhận diện hiệu ứng riêng của kiến trúc. Tỷ lệ kiểm duyệt theo nguồn được công bố trong đầu ra hỗ trợ; phép chấm vẫn có điều kiện trên sự kiện quan sát được, chưa xử lý bằng mô hình sống còn. Không có khoảng mới cho GRU hoặc các seed bổ sung trong đợt này.','']
 (OUT/'supplement_addition.md').write_text('\n'.join(parts),encoding='utf-8')
 rows=[]
 for t in TARGETS:
  f=agg.loc[agg.endpoint.eq(t)].sort_values('primary');rows.append(dict(endpoint=t,ordered_systems=json.loads(f[['system','primary','primary_seed_sd','mae','rmse']].to_json(orient='records'))))
 report=['# NFR1 — đối chứng khớp lại trên 12 nguồn','', '## Phạm vi','', '- 1.188 fit nội bộ; 651 fit ngoài; 2.870 epoch neural ngoài; 12/12 nguồn đã audit.', '- 864 tổ hợp nguồn–hệ thống–nhãn–seed được chấm; calibration không được đọc.', '- Hậu nghiệm; năm seed refit có điều kiện trên một seed chọn nội bộ.','', '## Sai số chính','',primary,'', '## Đối chiếu','',sensitivity,'', '## Đơn vị gốc','',units,'', '## Giới hạn','', '- SoH dùng MAE; RUL dùng RMSLE làm chỉ số chính. RMSE là trung bình RMSE từng cell, không phải pooled RMSE.', '- Không chọn lại ứng viên, epoch, seed; không khớp thêm, không thêm khoảng dự báo.', '- SNL exclusion chỉ đổi trọng số tổng hợp; không loại riêng proxy cell hoặc huấn luyện lại.', '- Không dùng p-value hoặc khoảng tin cậy suy diễn theo nguồn độc lập.', '- Dữ liệu kiểm duyệt không được coi là sai số 0; CALB/RUL90 là NE.', '- Historical và đối chứng mới có vai trò khác nhau; không ghép các bảng vào cùng một lượt thực nghiệm.','', '## Tệp','', '- `protocol.frozen.json`, `protocol_seal.json`: niêm phong trước lượt chấm này.', '- `score_manifest.json`: hash đầu vào/đầu ra, môi trường và giới hạn.', '- `source_seed_metrics.csv`, `source_summary.csv`: theo nguồn và seed.', '- `aggregate_metrics.csv`, `seed_macro_metrics.csv`: trung bình và độ lệch chuẩn.', '- `support.csv`: miền chấm và số cell bị kiểm duyệt theo nguồn.', '- `paired_source_deltas.csv`, `pairwise_summary.csv`, `leave_one_source_out.csv`, `snl_exclusion.csv`: đối chiếu và độ nhạy.', '- `cell_metrics.private.parquet`: thống kê cell, chỉ lưu local; chưa được phép công bố.','']
 (OUT/'assessment.md').write_text('\n'.join(report),encoding='utf-8')
 score.write(OUT/'rankings_descriptive.json',dict(rows=rows,rankings_not_new_model_selection=True))
 plot(seed)
 files=['assessment.md','supplement_addition.md','report_contrasts.json','rankings_descriptive.json']
 score.write(OUT/'report_manifest.json',dict(at_utc=score.now(),score_manifest=score.binding(OUT/'score_manifest.json'),source=score.binding(Path(__file__)),outputs={n:score.binding(OUT/n) for n in files},new_label_reads=False,new_training=False))
 print(agg[['system','endpoint','primary','primary_seed_sd','mae','rmse']].to_string(index=False));print(pd.DataFrame(chosen).to_string(index=False))
def plot(seed):
 sys.path.insert(0,str(ROOT/'.figure_runtime'));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.audit_tools/matplotlib_config'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 sources=sorted(seed.source.unique());fig,axes=plt.subplots(1,3,figsize=(12,6),sharey=True)
 colors={'TCN':'#2864a0','GRU':'#c45634'};names=['SoH30: MAE','RUL90: RMSLE','RUL80: RMSLE'];records=[]
 for ax,t,title in zip(axes,TARGETS,names):
  for system,offset in [('TCN',-.13),('GRU',.13)]:
   for y,source in enumerate(sources):
    a=seed.loc[seed.source.eq(source)&seed.endpoint.eq(t)&seed.system.eq(system)&seed.status.eq('EVALUABLE')].set_index('seed');b=seed.loc[seed.source.eq(source)&seed.endpoint.eq(t)&seed.system.eq('Extra Trees')&seed.status.eq('EVALUABLE')].set_index('seed')
    if a.empty:continue
    score.need(set(a.index)==set(b.index),'PLOT_PAIRED_SEEDS');d=a.primary-b.primary;mean=float(d.mean());low=float(d.min());high=float(d.max());ax.errorbar(mean,y+offset,xerr=[[mean-low],[high-mean]],fmt='o',ms=3.5,color=colors[system],elinewidth=1,capsize=2,label=system if y==0 else None);records.append(dict(system=system,endpoint=t,source=source,mean=mean,seed_min=low,seed_max=high))
  ax.axvline(0,color='#555555',lw=.8);ax.set_title(title,fontsize=11);ax.set_xlabel('Error difference vs Extra Trees',fontsize=9);ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
 axes[0].set_yticks(np.arange(len(sources)),sources);axes[0].invert_yaxis();axes[0].legend(frameon=False,loc='lower left',fontsize=9);fig.suptitle('Selected outer refits: source means and ranges across five paired seeds',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));folder=ROOT/'paper/latex/figures';stem=folder/'figS01_nfr1_source_deltas'
 for ext in ['pdf','svg','png']:fig.savefig(str(stem)+'.'+ext,dpi=180,bbox_inches='tight')
 plt.close(fig)
 score.write(OUT/'figure_manifest.json',dict(source_metrics=score.binding(OUT/'source_seed_metrics.csv'),rows=records,range='min-max of paired seed deltas; not confidence interval',files={ext:score.binding(Path(str(stem)+'.'+ext)) for ext in ['pdf','svg','png']}))
if __name__=='__main__':main()
