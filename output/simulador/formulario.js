const opcoes = labels => labels.map((label, i) => [i + 1, label]);
const simNao = [[0, "Não"], [1, "Sim"]];
// Percentis acumulados de _INCOMG1 na base bruta, entre respostas válidas:
// 5,53%; 14,50%; 25,60%; 39,30%; 70,17%; 92,28%.
// Quantis correspondentes da renda domiciliar habitual anualizada (VD5007 × 12)
// dos adultos da PNAD Contínua 2023 (1ª visita), ponderados por V1032.
const limitesRendaReais = [10800, 16824, 26400, 35976, 68400, 160956];

const formatoReais = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL", maximumFractionDigits: 0 });

function rotuloFaixa(inicio, fim, formatar) {
  if (inicio === undefined) return "Menos de " + formatar(fim);
  if (fim === undefined) return formatar(inicio) + " ou mais";
  return formatar(inicio) + " a menos de " + formatar(fim);
}

const faixasRenda = Array.from({ length: limitesRendaReais.length + 1 }, (_, i) => {
  const inicio = limitesRendaReais[i - 1];
  const fim = limitesRendaReais[i];
  return [i + 1, rotuloFaixa(inicio, fim, valor => formatoReais.format(valor))];
});
const grupos = {
  "Perfil": {
    Age: ["Qual é a faixa etária?", opcoes(["18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "80 ou mais"])],
    Sex: ["Qual é o sexo registrado?", [[0, "Masculino"], [1, "Feminino"]]],
    Education: ["Qual é a escolaridade?", opcoes(["Ensino médio incompleto", "Ensino médio completo", "Cursou faculdade ou curso técnico", "Concluiu faculdade ou curso técnico"])],
    Income: ["Qual é a faixa de renda anual total do seu domicílio?", faixasRenda],
  },
  "Histórico e saúde atual": {
    Peso: ["Qual é o seu peso em quilogramas?", {min:0.01,step:0.01}],
    Altura: ["Qual é a sua altura em centímetros?", {min:0.01,step:0.01}],
    HighBP: ["Já recebeu informação de que tem pressão alta?", simNao],
    HighChol: ["Já recebeu informação de que tem colesterol alto?", simNao],
    HeartDiseaseorAttack: ["Já teve doença coronariana ou infarto?", simNao],
    Stroke: ["Já teve AVC?", simNao],
    DiffWalk: ["Tem dificuldade séria para caminhar ou subir escadas?", simNao],
    GenHlth: ["Como avalia sua saúde geral?", opcoes(["Excelente","Muito boa","Boa","Regular","Ruim"])],
    MentHlth: ["Nos últimos 30 dias, em quantos dias você sentiu que sua saúde mental não estava bem?", {min:0,max:30,step:1},
      "Considere dias em que você se sentiu mal por estresse, tristeza ou outros problemas emocionais. Informe um número de 0 a 30. Se não houve nenhum dia, digite 0."],
    PhysHlth: ["Nos últimos 30 dias, em quantos dias você sentiu que sua saúde física não estava bem?", {min:0,max:30,step:1},
      "Considere dias em que você se sentiu mal por doença, dor ou lesão. Informe um número de 0 a 30. Se não houve nenhum dia, digite 0."],
  },
  "Hábitos e acesso à saúde": {
    PhysActivity: ["Fez atividade física fora do trabalho nos últimos 30 dias?", simNao],
    Smoker: ["Já fumou pelo menos 100 cigarros na vida?", simNao],
    HasHealthcare: ["Tem alguma cobertura de saúde?", simNao],
    NoDocbcCost: ["Deixou de consultar um médico por custo nos últimos 12 meses?", simNao],
    CholCheck: ["Fez exame de colesterol nos últimos cinco anos?", simNao],
  },
};

const perguntas = document.getElementById("perguntas");
for (const [titulo, campos] of Object.entries(grupos)) {
  const grupo = document.createElement("fieldset");
  const legenda = document.createElement("legend");
  legenda.textContent = titulo;
  grupo.append(legenda);
  const grade = document.createElement("div");
  grade.className = "grid";
  for (const [nome, [pergunta, config, ajuda]] of Object.entries(campos)) {
    const label = document.createElement("label");
    label.textContent = pergunta;
    const campo = document.createElement(Array.isArray(config) ? "select" : "input");
    campo.name = nome;
    campo.required = true;
    if (Array.isArray(config)) {
      campo.add(new Option("Selecione", "", true, true));
      for (const [valor, texto] of config) campo.add(new Option(texto, valor));
    } else {
      campo.type = "number";
      Object.assign(campo, config);
    }
    label.append(campo);
    if (ajuda || nome === "Income") {
      const nota = document.createElement("small");
      nota.id = `ajuda-${nome}`;
      nota.textContent = ajuda || "Some a renda mensal habitual dos moradores e multiplique por 12. Faixas aproximadas com base na PNAD Contínua 2023.";
      campo.setAttribute("aria-describedby", nota.id);
      label.append(nota);
    }
    grade.append(label);
  }
  grupo.append(grade);
  perguntas.append(grupo);
}
const form = document.getElementById("perfil");
const resultado = document.getElementById("resultado");
const botaoComparar = document.getElementById("comparar");
const comparacao = document.getElementById("comparacao");
const classes = ["Sem diabetes", "Pré-diabetes", "Diabetes"];
let ultimaEntrada;

function mostrarComparacao(aberta) {
  comparacao.hidden = !aberta;
  botaoComparar.setAttribute("aria-expanded", String(aberta));
  botaoComparar.textContent = aberta ? "Ocultar outros modelos" : "Ver outros modelos";
}

botaoComparar.addEventListener("click", () => {
  if (comparacao.hidden) {
    for (const [nome, arvores] of Object.entries(MODELOS_COMPARACAO)) {
      document.getElementById(`classe-${nome}`).textContent = classes[preverComArvores(arvores, ultimaEntrada)];
    }
  }
  mostrarComparacao(comparacao.hidden);
});

form.addEventListener("submit", event => {
  event.preventDefault();
  const respostas = new FormData(form);
  const altura = Number(respostas.get("Altura")) / 100;
  const imc = Math.round(Number(respostas.get("Peso")) / (altura ** 2) * 100) / 100;
  if (!Number.isFinite(imc) || imc < 12 || imc > 70) {
    form.elements.Peso.setCustomValidity("O IMC calculado está fora da faixa usada pelo modelo (12 a 70). Confira o peso e a altura.");
    form.elements.Peso.reportValidity();
    return;
  }
  const valores = MODELO.features.map(nome => {
    if (nome === "BMI") return imc;
    return Number(respostas.get(nome));
  });
  ultimaEntrada = valores;
  mostrarComparacao(false);
  document.getElementById("classe").textContent = classes[preverClasse(MODELO, valores)];
  resultado.hidden = false;
  resultado.focus();
});
function limparResultado() {
  form.elements.Peso.setCustomValidity("");
  ultimaEntrada = undefined;
  mostrarComparacao(false);
  resultado.hidden = true;
}
form.addEventListener("input", limparResultado);
form.addEventListener("reset", limparResultado);
