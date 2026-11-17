function preverClasse(modelo, valores) {
  const votos = [0, 0, 0];
  for (const [atributo, limite, esquerda, direita, peso] of modelo.trees) {
    const classe = atributo < 0 || Math.fround(valores[atributo]) <= limite ? esquerda : direita;
    votos[classe] += peso;
  }
  return votos.indexOf(Math.max(...votos));
}

function preverComArvores(arvores, valores) {
  const totais = [0, 0, 0];
  for (const arvore of arvores) {
    let no = arvore[0];
    while (no.length === 4) {
      const [atributo, limite, esquerda, direita] = no;
      no = arvore[Math.fround(valores[atributo]) <= limite ? esquerda : direita];
    }
    for (let i = 0; i < totais.length; i++) totais[i] += no[i];
  }
  return totais.indexOf(Math.max(...totais));
}

if (typeof module !== "undefined") module.exports = { preverClasse, preverComArvores };
