import source

arquivos = source.lerPasta()

nome_do_arquivo = 'resultado_gabaritos.txt'

with open(nome_do_arquivo, "w") as arquivo:
    for imagem in arquivos:
        entrada = ''
        img, respostas = source.cortarGabarito(f'Imagens/{imagem}')

        #source.mostrarImagem(img)
        
        entrada = ' '.join(respostas[alternativa] for alternativa in range(1, 51))

        acertos = source.compararRespostas(respostas)

        arquivo.write(f'{imagem} - Acertos: {acertos}/50 - Respostas: {entrada}\n')
print(f'Leitura concluida! Verifique o resultado no arquivo {nome_do_arquivo}')