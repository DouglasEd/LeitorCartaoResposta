import source

resultados=''
arquivos = source.lerPasta()
for imagem in arquivos:
    img , respostas = source.cortarGabarito(f'Imagens/{imagem}')
    acertos= source.compararRespostas(respostas)
    print(f'Acertos: {acertos}/50')
    resultados+=f'Acertos: {acertos}/50\n'

if input('Voce gostaria de salvar os resultados em um arquivo [S/N] ').upper() == 'S':
    nome_do_arquivo=input('Escreva o nome do arquivo:') + '.txt'
    with open(nome_do_arquivo, "w") as arquivo:
        arquivo.write(resultados)
