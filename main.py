import source

arquivos = source.lerPasta()
for imagem in arquivos:
    resultados=''
    entrada=''
    img , respostas = source.cortarGabarito(f'Imagens/{imagem}')

    for alternativa in range(1,51):
        entrada += respostas[alternativa] + '\n'
    
    acertos= source.compararRespostas(respostas)

    resultados=f'Acertos: {acertos}/50\n'

    if input('Voce gostaria de salvar os resultados em um arquivo [S/N] ').upper() == 'S':
        nome_do_arquivo=input('Escreva o nome do arquivo:') + '.txt'
        with open(nome_do_arquivo, "w") as arquivo:
            arquivo.write(resultados+'\n'+ entrada)
    resultados=''
