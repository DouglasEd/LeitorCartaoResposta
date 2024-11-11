import LB

img_path = 'Imagens/cartao1-ideal.jpg'
#cartao1-ideal.jpg
#cartao2-falha.jpg
#cartao3-torto.jpg

img , Respostas = LB.CortarGabarito(img_path)
acertos= LB.CompararRespostas(Respostas)
print(f"Esse poha acertou {acertos}/50")
LB.ShowImg(img)