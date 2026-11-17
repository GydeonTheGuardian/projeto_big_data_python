from pathlib import Path
import ast
import re
from html import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

root = Path("outputs")
log = (root / "execucao_final_modelagem.txt").read_text(encoding="utf-8-sig")
assert "PIPELINE DE MODELAGEM PREDITIVA CONCLUÍDO COM SUCESSO!" in log
models = ("Decision Tree", "Random Forest", "AdaBoost")
classes = ("Sem diabetes", "Pré-diabetes", "Diabetes")
totals = (43880, 1341, 8174)
blocks = log.split("AVALIAÇÃO DO MODELO: ")[1:]
searches = re.findall(r"Melhores hiperparâmetros: (.+)\s+F1 macro médio na validação: ([\d.]+)", log)
assert len(blocks) == len(searches) == len(models)
records = []
for name, block, (params, cv) in zip(models, blocks, searches):
    matrix_text = block.split("2. MATRIZ DE CONFUSÃO")[1].split("3. IMPORTÂNCIA")[0]
    matrix = [[int(v.replace(",", "")) for v in re.findall(r"([\d,]+)\s+\(", line)]
              for line in matrix_text.splitlines()
              if re.match(r"\s*(Não Diabético|Pré-diabético|Diabético)\s*\|", line)]
    assert tuple(map(sum, matrix)) == totals
    recall = [matrix[i][i] / totals[i] for i in range(3)]
    precision = [matrix[i][i] / max(1, sum(row[i] for row in matrix)) for i in range(3)]
    f1 = sum(2*p*r/(p+r) if p+r else 0 for p,r in zip(precision, recall)) / 3
    accuracy = sum(matrix[i][i] for i in range(3)) / sum(totals)
    config = ", ".join(f"{key.removeprefix('modelo__')}={value}" for key,value in ast.literal_eval(params).items())
    records.append((name,float(cv),f1,accuracy,precision[1],recall[1],config,matrix))

def decimal(value):
    return f"{value:.4f}".replace(".", ",")
def percent(value):
    return f"{value:.2%}".replace(".", ",")
def integer(value):
    return f"{value:,}".replace(",", ".")

headers = ["Modelo", "F1 CV", "F1 teste", "Acurácia", "Precisão¹", "Sensibilidade¹"]
rows = [[n,decimal(cv),decimal(f1),percent(a),percent(p),percent(r)] for n,cv,f1,a,p,r,_,_ in records]
pdf = Path("output/pdf/resultados_modelagem_diabetes.pdf")
pdf.parent.mkdir(parents=True,exist_ok=True)
tmp_pdf = pdf.with_suffix(".tmp")
c = canvas.Canvas(str(tmp_pdf),pagesize=A4)
c.setTitle("Avaliação final dos modelos de diabetes")
ink = "#182C42"
blue = "#176D74"

def txt(x,y,message,size=10,bold=False,color=ink):
    c.setFillColor(colors.HexColor(color))
    c.setFont("Helvetica-Bold" if bold else "Helvetica",size)
    c.drawString(x,y,message)

def title(kicker, heading, page):
    txt(44,783,kicker,10,True,blue)
    txt(44,747,heading,27,True)
    txt(44,42,"Tópicos em Big Data em Python",9)
    txt(510,42,f"{page} / 2",9)

def table(y, headings, values, widths, matrix=False):
    for index,row in enumerate([headings,*values]):
        x=44
        for column,(value,width) in enumerate(zip(row,widths)):
            accent=matrix and index == column and column > 0
            fill=ink if index == 0 else "#D5ECE9" if accent else "#EDF3F7" if index%2 else "#F8FAFC"
            c.setFillColor(colors.HexColor(fill))
            c.rect(x,y-8,width,24,stroke=0,fill=1)
            txt(x+8,y,str(value),9,index==0 or column==0 or accent,"#FFFFFF" if index==0 else ink)
            x+=width
        y-=24

title("CLASSIFICAÇÃO MULTICLASSE", "Avaliação final dos modelos", 1)
txt(44,705,"Base tratada com 266.972 registros e 18 atributos preditores.")
txt(44,687,"Divisão estratificada de 80% para treino (213.577) e 20% para teste (53.395).")
txt(44,669,"Grid Search com 3 folds estratificados, F1 macro e semente 42.")
txt(44,651,"Oversampling somente no treino de cada fold e no retreino final.")
txt(44,633,"Cada classe minoritária foi elevada à metade da classe majoritária.")
txt(44,595,"Validação cruzada e teste reservado",15,True)
table(565,headers,rows,[100,67,67,77,83,113])
txt(44,467,"¹ Métricas da classe pré-diabetes. F1 é macro nos dois conjuntos.",9)
txt(44,423,"Hiperparâmetros selecionados",15,True)
for i,(n,_,_,_,_,_,params,_) in enumerate(records):
    y=393-i*45
    txt(44,y,n,11,True,blue)
    txt(44,y-17,params,10)
best=max(records,key=lambda r:r[2])
best_recall=max(records,key=lambda r:r[5])
txt(44,232,"Leitura dos resultados",15,True)
txt(44,209,f"{best[0]} obteve o maior F1 macro no teste ({decimal(best[2])}).")
txt(44,190,f"{best_recall[0]} teve a maior sensibilidade para pré-diabetes ({percent(best_recall[5])}).")
txt(44,171,"A precisão indica a proporção de acertos entre as previsões de pré-diabetes.")
c.showPage()
title("RESULTADOS NO TESTE RESERVADO","Matrizes de confusão",2)
txt(44,710,"Linhas representam as classes reais e colunas representam as classes previstas.")
for i,(n,_,_,_,_,_,_,matrix) in enumerate(records):
    y=665-i*173
    txt(44,y,n,15,True,blue)
    table(y-29,["Classe real",*classes,"Total"],[[label,*map(integer,row),integer(sum(row))]
          for label,row in zip(classes,matrix)],[117,107,107,98,78],matrix=True)
txt(44,137,"As células destacadas representam classificações corretas.",10)
c.save()
tmp_pdf.replace(pdf)

def html_table(head, body):
    return "<table><thead><tr>"+"".join(f'<th scope="col">{escape(str(v))}</th>' for v in head)+"</tr></thead><tbody>"+ "".join(
        "<tr>"+f'<th scope="row">{escape(str(row[0]))}</th>'+"".join(f"<td>{escape(str(v))}</td>" for v in row[1:])+"</tr>" for row in body)+"</tbody></table>"

details = "".join(f'<details><summary>{n}</summary><p><code>{escape(config)}</code></p>'+html_table(
    ["Classe real / prevista",*classes],[[label,*map(integer,row)] for label,row in zip(classes,matrix)])+"</details>"
    for n,_,_,_,_,_,config,matrix in records)
html=Path("output/desempenho_modelos.html")
chart = "<ul>"+ "".join(f'<li>{n} <meter min="0" max="1" value="{f1}" aria-label="F1 macro de {n}"></meter> {decimal(f1)}</li>'
                        for n,_,f1,*_ in records)+"</ul>"
html.write_text("""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Desempenho dos modelos</title>
<style>body{font:1rem/1.55 system-ui,sans-serif;color:#182c42;background:#f6f8fb;margin:0}main{max-width:880px;margin:auto;padding:32px 16px}h1{line-height:1.1}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:right;border-bottom:1px solid #cbd7e1}th:first-child{text-align:left}thead{background:#182c42;color:white}details{margin-block:12px;background:white;padding:12px}summary{font-weight:650;cursor:pointer}.scroll{overflow-x:auto}code{overflow-wrap:anywhere}</style>
<main><h1>Desempenho dos modelos</h1><p>Grid Search com 3 folds estratificados e seleção por F1 macro. Avaliação final em 53.395 registros reservados.</p><div class="scroll">"""
 +html_table(headers,rows)+"""</div><p>¹ Precisão e sensibilidade para pré-diabetes.</p><h2>F1 macro no teste</h2>"""
 +chart+"""<h2>Matrizes de confusão</h2><p>Abra um modelo para ver os hiperparâmetros e a matriz de confusão. As linhas mostram as classes reais.</p>"""
+details+"</main></html>",encoding="utf-8")
for row in rows:
    print(row)
print(pdf,html)
