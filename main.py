import LB

img_path = 'cartao1-ideal.jpg'
Resultados=''
Arquivos = LB.LerPasta()
for imagem in Arquivos:
    img , Respostas = LB.CortarGabarito(f'Imagens/{imagem}')
    acertos= LB.CompararRespostas(Respostas)
    print(f'Acertos: {acertos}/50')
    Resultados+=f'Acertos: {acertos}/50\n'
if input('Voce gostaria de salvar so resultados em um arquivo [S/N] ').upper() == 'S':
    NomeArq=input('Escreva o nome do arquivo:')
    with open(NomeArq, "w") as arquivo:
        arquivo.write(Resultados)
