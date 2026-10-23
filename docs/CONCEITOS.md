# Conceitos do módulo 1, explicados pelo código

Guia de estudo em português. Cada conceito aponta para o arquivo onde ele acontece de verdade.

## 1. IA, Machine Learning, Deep Learning e LLM

- **IA** é o guarda-chuva: qualquer sistema que faz algo que parece "inteligente", inclusive regras escritas à mão (`if preço > 100 então frete grátis`).
- **Machine Learning** é quando, em vez de escrever a regra, você dá exemplos e o programa descobre a regra sozinho. Este projeto é isso: ninguém escreveu "está dentro se a distância for menor que 0,6", a rede descobriu.
- **Deep Learning** é ML usando redes neurais com várias camadas. A rede do círculo (2 camadas escondidas) já é uma rede "profunda" de brinquedo.
- **LLM** (Claude, GPT, Gemini) é Deep Learning em escala gigante: a mesma ideia de pesos + perda + gradiente, com bilhões de pesos, treinada para prever a próxima palavra.

## 2. O neurônio

Um neurônio faz duas coisas (veja `forward()` em `src/network.ts`):

1. **Soma ponderada:** `z = peso1·entrada1 + peso2·entrada2 + ... + bias`
2. **Ativação:** passa `z` por uma função não linear, como `tanh(z)`.

Os **pesos** dizem quanto cada entrada importa. O **bias** desloca o resultado. Aprender é só ajustar esses números.

## 3. Por que precisa de ativação (e de mais de uma camada)

Sem ativação, empilhar camadas é inútil: uma soma de somas continua sendo uma soma, ou seja, uma reta. E nenhuma reta separa o XOR:

```
(0,1)=1   (1,1)=0
(0,0)=0   (1,0)=1
```

Com `tanh` na camada escondida, a rede consegue "dobrar" o espaço e separar os pontos. Por isso o XOR é o "hello world" das redes neurais. Ativações em `src/activations.ts`.

## 4. Tensores

Tensor é só o nome genérico para "array de números com várias dimensões":

- escalar (0D): `0.5`
- vetor (1D): uma entrada `[0.3, -0.8]`
- matriz (2D): os pesos de uma camada `weights[j][i]`
- 3D ou mais: um lote de imagens, por exemplo `[imagens][linhas][colunas]`

Aqui usamos arrays comuns de JavaScript. O TensorFlow.js faz a mesma conta, só que em GPU e muito mais rápido.

## 5. Função de perda (loss)

Um número que mede o quanto a rede errou. Treinar é diminuir esse número. Em `src/losses.ts`:

- **MSE** (erro quadrático médio): média de `(previsto - esperado)²`. Bom para prever valores (preço, nota).
- **Binary cross-entropy:** para sim/não. Pune muito quando a rede está *confiante e errada* (diz 99% e era 0).

## 6. Backpropagation: o coração do aprendizado

Pergunta: "se eu mexer um pouquinho neste peso, o erro sobe ou desce, e quanto?". A resposta é o **gradiente** daquele peso.

O backpropagation calcula isso para todos os pesos de uma vez, de trás para frente (veja `gradients()` em `src/network.ts`):

1. Na saída: quanto o erro muda se a previsão mudar (derivada da perda).
2. Volta uma camada: distribui esse "culpa" para os pesos, proporcional ao quanto cada um contribuiu (regra da cadeia do cálculo).
3. Repete até a primeira camada.

**Como saber se está certo?** O teste em `tests/network.test.ts` faz a conta do jeito "burro": mexe cada peso um pouquinho para cima e para baixo e mede o erro de verdade. Se bate com o backprop, a matemática está correta. Isso se chama *gradient check*.

## 7. Gradiente descendente

Com os gradientes em mãos, cada peso anda um passinho na direção contrária (veja `step()`):

```
peso = peso - taxaDeAprendizado × gradiente
```

Imagine descer uma montanha no escuro, sentindo a inclinação do chão com o pé.

- **Taxa de aprendizado (learning rate):** o tamanho do passo. Grande demais, você pula o vale e o erro oscila. Pequeno demais, demora uma eternidade. Teste mudar `learningRate` em `examples/xor.ts` para `5` e para `0.01`.
- **Época (epoch):** uma passada completa por todos os dados.
- **Mini-batch:** em vez de atualizar depois de cada exemplo ou só depois de todos, atualiza a cada lote (16 no exemplo do círculo). É o meio-termo usado na prática.

## 8. O ciclo completo: dados → treino → validação → inferência

Está inteiro em `examples/circle.ts`:

1. **Dados:** 600 pontos, com 5% de rótulos errados de propósito, porque dado real nunca é limpo.
2. **Separação treino/validação:** 80% para treinar, 20% guardados. A rede nunca vê esses 20% durante o treino.
3. **Treino:** a curva de perda cai.
4. **Validação:** a acurácia nos dados guardados é a nota que importa.
5. **Inferência:** usar a rede treinada em pontos novos (`predict`).

**Overfitting:** se o treino dá 100% e a validação 70%, a rede decorou em vez de aprender. Aqui, como 5% dos rótulos estão errados, 100% no treino seria sinal de decoreba. Ficar perto de 92% nos dois é o resultado saudável.

## 9. Ligação com LLMs (o que vem nos próximos módulos)

| Aqui | Num LLM |
| --- | --- |
| Entrada `[x, y]` | Texto quebrado em **tokens** |
| Pesos de uma camada | Bilhões de pesos em camadas de **transformer** |
| Camada escondida aprende "distância do centro" | Camadas aprendem **embeddings**: vetores que representam significado |
| Perda: acertou dentro/fora do círculo? | Perda: acertou a **próxima palavra**? |
| Treina em segundos | Treina por meses em milhares de GPUs |

Embeddings são a base do RAG (módulo 6): textos com significado parecido viram vetores próximos, e a busca semântica é só achar os vetores mais perto.

## 10. TensorFlow.js: a mesma rede, feita por uma biblioteca

No playground da demo dá para trocar o motor entre "From scratch" e "TensorFlow.js" (veja `web/src/engines.ts`). As duas versões montam exatamente a mesma arquitetura e treinam do mesmo jeito (SGD com binary cross-entropy). O que muda:

| Nossa rede | TensorFlow.js |
| --- | --- |
| `NeuralNetwork` com arrays | `tf.sequential()` com `tf.layers.dense()` |
| `gradients()` escrito à mão | Gradientes calculados automaticamente (*autodiff*) |
| Laços em JavaScript | Operações em **tensores**, rodando na GPU via WebGL |
| `train()` | `model.fit()` |

Depois de escrever o backprop na mão, o `model.fit()` deixa de ser mágica: ele faz o que o nosso `train()` faz, só que mais rápido e para qualquer arquitetura.

**Detalhe de front-end:** o TensorFlow.js tem 1,3 MB. Ele é carregado com `import()` dinâmico só quando alguém escolhe esse motor, então a página abre com 14 KB de JavaScript. Isso se chama *code splitting*.

## 11. Overfitting ao vivo

No playground, treine o círculo por mais de mil épocas e observe o gráfico: a perda de treino continua caindo, mas a de validação começa a **subir**. A fronteira fica cheia de "ilhas" em volta dos pontos com rótulo errado. A rede parou de aprender a forma e começou a decorar o ruído. A página avisa quando detecta isso (`isOverfitting()` em `web/src/playground.ts`).

Como se combate overfitting na prática: parar o treino quando a validação piora (*early stopping*), usar uma rede menor, conseguir mais dados ou usar *dropout* (desligar neurônios ao acaso durante o treino, que usamos no classificador da webcam).

## 12. Transfer learning e embeddings de imagem

Treinar um modelo de visão do zero exige milhões de fotos e dias de GPU. O classificador da webcam (`web/src/webcam.ts`) usa um atalho:

1. O **MobileNet** já foi treinado no ImageNet (1,2 milhão de imagens, 1.000 categorias). Pedimos a ele não a categoria final, mas a camada anterior: um **embedding** de 1.280 números que descreve a imagem (bordas, formas, texturas, objetos).
2. Enquanto você segura o botão, cada frame vira um embedding e é guardado com o rótulo da classe.
3. Uma rede pequena (1.280 → 64 → número de classes) aprende a separar esses embeddings. Com poucas dezenas de exemplos ela treina em segundos.

É a mesma ideia que você vai usar com LLMs: aproveitar um modelo grande pré-treinado e só adaptar a última parte. O fine-tuning do módulo 10 é uma versão mais profunda disso.

## Exercícios para fixar

1. Em `examples/xor.ts`, troque a camada escondida de 4 para 1 neurônio. O que acontece com a perda? Por quê?
2. Troque `"tanh"` por `"relu"`. Converge mais rápido ou mais devagar?
3. Em `examples/circle.ts`, mude o ruído de `0.05` para `0.3`. O que acontece com a acurácia de validação?
4. Rode `npm test` depois de "estragar" de propósito uma linha do backprop (por exemplo, troque `-=` por `+=` no `step()`). Veja o teste pegando o erro.
5. No playground, escolha a **Espiral** com camadas "4". A rede consegue? Agora tente "24 → 24 → 24". O que a quantidade de neurônios muda?
6. Ainda no playground, coloque learning rate **3**. O que acontece com a perda e com o mapa?
7. Na webcam, treine "Thumbs up" com a mão em um só lugar da tela e depois mostre a mão em outro canto. O modelo erra? Por que dados variados importam?
