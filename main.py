import LB

img_path = 'Imagens/cartao3-torto.jpg'
#cartao1-ideal.jpg
#cartao2-falha.jpg
#cartao3-torto.jpg

img , Respostas = LB.CortarGabarito(img_path)
print(Respostas)
LB.ShowImg(img)