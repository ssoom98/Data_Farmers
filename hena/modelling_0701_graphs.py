
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import xlsxwriter

# 한글 폰트 설정 (맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 데이터 불러오기
file_path = r"C:\price_prediction\scaffold\outputs_0701\all_products_price_forecast.csv"
df = pd.read_csv(file_path)

# '거래일자'를 datetime으로 변환
df['거래일자'] = pd.to_datetime(df['거래일자'])

# 일요일 데이터 삭제
df = df[df['거래일자'].dt.weekday != 6]

# 군집 딕셔너리
cluster_dict = {
    '깐양파':2, '깐쪽파':0, '깻잎(일반)':4, '노랑파프리카':3, '녹광':3, '뉴그린':4, '느타리버섯(일반)':1,
    '단호박':6, '당근(일반)':2, '대추방울':5, '대파(일반)':2, '돌미나리':4, '레드치커리':4, '만생양파':2,
    '맛느타리버섯':1, '미나리(일반)':4, '밤고구마':6, '방울토마토':5, '백다다기':4, '브로코리(국산)':4,
    '빨강파프리카':3, '새송이(일반)':1, '수미':6, '시금치(일반)':4, '애호박':4, '양배추(일반)':5,
    '양상추(일반)':4, '양송이(일반)':1, '영양부추':4, '오이맛고추':3, '완숙토마토':5, '일반부추':4,
    '자주양파':2, '적채(일반)':5, '쥬키니호박':6, '쪽파(일반)':0, '청양':3, '청초(일반)':3, '청피망':3,
    '취청':4, '치커리(일반)':4, '치콘':4, '토마토(일반)':5, '팽이':1, '포장쪽파':0, '표고버섯(일반)':1,
    '호박고구마':6, '홍고추(일반)':3, '홍청양':3, '홍피망':3
}

# 색상 팔레트 (농산물 개수 대비 충분히 확보)
colors = plt.cm.tab20.colors

# 결과 저장할 엑셀 파일
output_path = r"C:\price_prediction\scaffold\outputs_0701\graph_0701.xlsx"
writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

# 군집별 처리
for cluster_id in sorted(set(cluster_dict.values())):
    # 해당 군집의 농산물 리스트
    products = [prod for prod, cid in cluster_dict.items() if cid == cluster_id]
    
    # 색상 매핑
    color_map = {prod: colors[i % len(colors)] for i, prod in enumerate(products)}
    
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(12, 6))
    for prod in products:
        if prod in df.columns:
            series = df[['거래일자', prod]].copy()
            series.loc[series[prod] <= 0, prod] = None  # 0 이하 제거
            ax.plot(series['거래일자'], series[prod], label=prod, color=color_map[prod])
    
    ax.set_title(f"군집 {cluster_id} 농산물 가격 예측")
    ax.set_xlabel("거래일자")
    ax.set_ylabel("가격")
    ax.legend()
    
    # 시트 이름
    sheet_name = f"Cluster_{cluster_id}"
    
    # 엑셀에 색상 매핑표 저장
    color_table = pd.DataFrame({
        '농산물': list(color_map.keys()),
        '색상(RGB)': [str(tuple(int(c*255) for c in color_map[p])) for p in color_map]
    })
    color_table.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    
    # 그래프를 엑셀 시트에 삽입
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]
    image_path = f"cluster_{cluster_id}.png"
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    worksheet.insert_image(len(color_table)+2, 0, image_path)

# 최종 저장
writer.close()

print("✅ 작업 완료: 군집별 그래프와 색상 매핑표가 graph_0701.xlsx에 저장되었습니다.")
