import pygame
from pygame.locals import *
from rede_neural_pong import Rede_Neural
from random import randint

def cor():
    return (randint(0,255), randint(0,255), randint(0,255))

print(cor())
# Inicialização do Pygame
pygame.init()


# Configurações da janela do jogo
largura_tela = 1000
altura_tela = 600
tela = pygame.display.set_mode((largura_tela, altura_tela))
pygame.display.set_caption('Pong')

# Cores
BRANCO = (255, 255, 255)

# Configurações da raquete
raquete_largura = 10
raquete_altura = 100
raquete_posicao_x = 20
raquete_posicao_y = altura_tela // 2 - raquete_altura // 2
raquete_velocidade = 10

# Configurações da bola
bola_tamanho = 10
bola_posicao_x = largura_tela // 2 - bola_tamanho // 2
bola_posicao_y = altura_tela // 2 - bola_tamanho // 2
bola_velocidade_x = 7
bola_velocidade_y = 7
rede = Rede_Neural(bola_posicao_x/largura_tela, bola_posicao_y/altura_tela, raquete_posicao_y/altura_tela, cor=cor())

# Loop principal do jogo
rodando = True
clock = pygame.time.Clock()
while rodando:
    clock.tick(60)
    # Lidar com eventos
    for event in pygame.event.get():
        teclas_pressionadas = pygame.key.get_pressed()

        if event.type == QUIT or teclas_pressionadas[K_q]:
            rodando = False

    
    # Movimentação da raquete
    #teclas_pressionadas = pygame.key.get_pressed()
    #if teclas_pressionadas[K_UP] and raquete_posicao_y > 0:
    #    raquete_posicao_y -= raquete_velocidade
    #if teclas_pressionadas[K_DOWN] and raquete_posicao_y + raquete_altura < altura_tela:
    #    raquete_posicao_y += raquete_velocidade

    # Movimentação da bola
    bola_posicao_x += bola_velocidade_x
    bola_posicao_y += bola_velocidade_y
    
    # Verificar colisões com a raquete
    if (
        bola_posicao_x <= raquete_posicao_x + raquete_largura
        and raquete_posicao_y <= bola_posicao_y <= raquete_posicao_y + raquete_altura
    ):
        bola_velocidade_x = abs(bola_velocidade_x)  # Inverter a direção da bola horizontalmente
        erro = 0


    decisao = rede.feedForward()
    if decisao > 0.52 and raquete_posicao_y > 0:
        raquete_posicao_y -= raquete_velocidade
    elif decisao < 0.48 and raquete_posicao_y + raquete_altura < altura_tela:
        raquete_posicao_y += raquete_velocidade

    # Verificar colisões com as paredes
    if bola_posicao_x >= largura_tela - bola_tamanho:
        bola_velocidade_x = -(bola_velocidade_x + 0.1)
    if bola_posicao_y <= 0 or bola_posicao_y >= altura_tela - bola_tamanho:
        bola_velocidade_y = -(bola_velocidade_y + 0.1)


    if bola_posicao_x <= raquete_posicao_x:
        bola_velocidade_x = -bola_velocidade_x
        erro = ((raquete_posicao_y + (raquete_altura / 2)) - (bola_posicao_y + (bola_tamanho/2))) / 10
        rede.backPropagation(erro)
        print("Bateu")

    erro = ((raquete_posicao_y + (raquete_altura / 2)) - (bola_posicao_y + (bola_tamanho/2))) / 100
    rede.backPropagation(erro)
    erro = 0

    rede.atualizarEntradas(bola_posicao_x/largura_tela, bola_posicao_y/altura_tela, raquete_posicao_y/altura_tela)
    
    # Limpar a tela
    tela.fill((0, 0, 0))  # Preencher a tela com a cor preta

    # Desenhar a raquete
    pygame.draw.rect(tela, rede.getCor(), (raquete_posicao_x, raquete_posicao_y, raquete_largura, raquete_altura))

    # Desenhar a bola
    pygame.draw.rect(tela, BRANCO, (bola_posicao_x, bola_posicao_y, bola_tamanho, bola_tamanho))

    # Atualizar a tela
    # rede.saida()
    
    pygame.display.update()

# Encerrar o Pygame
rede.salvarPesos()
pygame.quit()
