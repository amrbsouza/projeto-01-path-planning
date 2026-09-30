import heapq
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import math

class AStarPathfinder:
    def __init__(self, map_array: np.array, start: tuple, goal: tuple, wall_influence=5.0, buffer_factor=2.0):
        """
        Inicializa o A* com mapa, ponto inicial, objetivo e parâmetros de influência.

        Args:
            map_array (np.array): Mapa binário (obstáculos e caminho livre).
            start (tuple): Ponto inicial (linha, coluna).
            goal (tuple): Ponto objetivo (linha, coluna).
            wall_influence (float): Peso da proximidade das paredes.
            buffer_factor (float): Escala da influência das paredes.
        """
        self.start = start
        self.goal = goal
        self.wall_influence = wall_influence
        self.buffer_factor = buffer_factor
        self.GOAL_REACHEABLE = False  
        
        # Prepara o mapa, expandindo suas bordas e ajustando o array.
        self.map = map_array.copy()
        self.map_array = self.preprocess_map(map_array)

        # Cria um campo potencial baseado no mapa para influenciar o caminho.
        self.potential_field = self.create_potential_field()
        

    def preprocess_map(self, map_array: np.array) -> np.array: #
        """
        Ajusta o mapa, convertendo valores intermediários para obstáculos.

        Args:
            map_array (np.array): Mapa original.

        Returns:
            np.array: Mapa processado.
        """
        mapa_processado = map_array.copy()
        # 0: obstáculo - mantém o RGB preto do original
        mapa_processado[mapa_processado == 128] = 1 # posição desconhecida
        mapa_processado[mapa_processado == 255] = 2 # posição conhecida
        return mapa_processado

    def create_potential_field(self) -> np.array:
        """
        Gera campo potencial com base na distância de obstáculos.
        Penaliza muito mais aggressivamente pontos próximos a paredes.

        Returns:
            np.array: Campo potencial (alto perto de paredes, baixo longe).
        """
        distancia = distance_transform_edt(self.map_array != 0)
        # Penalidade inversamente proporcional ao quadrado da distância
        # Muito alta perto de paredes, cai rápido longe delas
        campo_potencial = (self.wall_influence ** 3) / np.maximum(distancia, 1) ** 2
        return campo_potencial

    def heuristic(self, a: tuple, b: tuple) -> float:
        """
        Calcula a heurística entre dois pontos com forte penalidade de parede.

        Args:
            a (tuple): Ponto A.
            b (tuple): Ponto B.

        Returns:
            float: Distância ao objetivo + forte penalidade de proximidade a parede.
        """
        # distance_to_goal = math.dist(a, b)
        # # Penalidade muito forte baseada no campo potencial
        # field_penalty = self.potential_field[a[0], a[1]] * 10
        # return distance_to_goal + field_penalty
        
        return math.dist(a, b)

    def find_path(self):
        """
        Executa o algoritmo A* para encontrar caminho até o objetivo.

        Returns:
            dict: Predecessores dos nós no caminho. Se o caminho não for encontrado, retorna None.
            tuple: O ponto final (objetivo) ou None se não encontrado.
        """
        start = self.start
        goal = self.goal

        # Estrutura A*
        open_set = []
        heapq.heappush(open_set, (0, start))

        # came_from eh o dicionário dos predecessores
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        open_set_hash = {start}

        # Movimentos permitidos: cima, baixo, esquerda e direita
        neighbors_moves = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            # Chegou no objetivo
            if current == goal:
                return came_from, current

            x, y = current

            for dx, dy in neighbors_moves:
                nx, ny = x + dx, y + dy

                # Verifica limites do mapa
                if nx < 0 or ny < 0 or nx >= self.map.shape[0] or ny >= self.map.shape[1]:
                    continue

                # Soh anda em área livre
                if self.map[nx, ny] == 0:
                    continue

                neighbor = (nx, ny)

                # Custo base (1) + penalidade forte por proximidade a parede
                base_cost = 1.0
                wall_penalty = self.potential_field[nx, ny] * 100
                tentative_g_score = g_score[current] + base_cost + wall_penalty

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        print("Caminho não encontrado")
        return None, None

    def reconstruct_path(self, came_from: dict, current: tuple) -> list:
        """
        Reconstrói o caminho a partir do ponto final até o inicial.
        
        Args:
            came_from (dict): O dicionário de predecessores no caminho.
            current (tuple): O ponto final (objetivo).
        
        Returns:
            list: Lista de tuplas com caminho reconstruído.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def know_path(self, path: list) -> list:
        """
        Remove trechos desconhecidos e ajusta o caminho, se necessário.

        Args:
            path (list): Caminho completo.

        Returns:
            list: Caminho ajustado.
        """
        if path is None:
            return None
        
        # Mantém apenas pontos em área conhecida (2).
        known_path = []
        for point in path:
            x, y = point
            if self.map_array[x, y] == 2:
                known_path.append(point)
        
        return known_path[:-3] if known_path else None

    def simplify_path(self, path: list) -> list:
        """
        Simplifica o caminho removendo direções repetidas.
        
        Args:
            path (list): Caminho completo.

        Returns:
            list: Caminho simplificado.
        """
        if path is None or len(path) <= 2:
            return path
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            current = path[i]
            prev = simplified[-1]
            next_point = path[i + 1]
            
            # Check if three points are collinear using cross product
            # If cross product is 0, they are collinear and middle point can be removed
            dx1 = current[0] - prev[0]
            dy1 = current[1] - prev[1]
            dx2 = next_point[0] - current[0]
            dy2 = next_point[1] - current[1]
            
            cross_product = dx1 * dy2 - dy1 * dx2
            
            # Keep the point if it's not collinear (not a straight line)
            if cross_product != 0:
                simplified.append(current)
        
        simplified.append(path[-1])
        return simplified

    def plot_path(self, path: list, simplified_path: list):
        """
        Exibe o mapa com o caminho completo e o simplificado.

        Args:
            path (list): O caminho completo encontrado.
            simplified_path (list): O caminho simplificado encontrado.
            valid_path (list): O caminho válido encontrado.
        """
        simplified_path = self.simplify_path(path)
        plt.figure(figsize=(10, 10))
        plt.imshow(self.map, cmap='gray')
        plt.scatter(self.start[1], self.start[0], color='green', s=100, label='Início')
        plt.scatter(self.goal[1], self.goal[0], color='blue', s=100, label='Objetivo')

        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, color='magenta', linewidth=1, label='Caminho Completo')
            simp_x, simp_y = zip(*simplified_path)
            plt.plot(simp_y, simp_x, color='red', linewidth=2, linestyle='--', label='Caminho Simplificado')
        else:
            plt.title("Caminho não encontrado")

        plt.legend()
        plt.axis('equal')
        plt.savefig('path.png')
        plt.show()

    def run(self, show_path=True):
        """
        Essa função é chamada pelo navegador para executar o algoritmo A* e gerar o caminho.
        Executa o processo completo: busca, reconstrução, simplificação e visualização do caminho.
        
        Args:
            show_path (bool): Se True, exibe o caminho graficamente.

        Returns:
            list or None: Caminho simplificado ou None se não encontrado.
        """
        print("Iniciando busca pelo caminho...")
        came_from, final_node = self.find_path()

        if final_node:
            print("Reconstruindo caminho...")
            path = self.reconstruct_path(came_from, final_node)
            
            print("Robo não anda no disconhecido")
            path = self.know_path(path)

            print("Caminho encontrado, simplificando...")
            simplified_path = self.simplify_path(path)

            print("Plotando o caminho...")
            if show_path:
                self.plot_path(path, simplified_path)
            
            return simplified_path
        else:
            print("Nenhum caminho pôde ser encontrado.")
            return None


def prep_map(map_path: str) -> np.array:
    """
    Prepara o mapa carregando e processando a imagem de entrada.

    Args:
        map_path (str): O caminho do arquivo do mapa.

    Returns:
        np.array: O mapa processado como um array numpy.
    """
    map_array = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    map_array[map_array == 0] = 0
    map_array[map_array == 205] = 128
    map_array[map_array == 254] = 255
    map_array[(map_array >= 60) & (map_array != 128) & (map_array != 255)] = 0
    map_array = map_array.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    map_array = cv2.morphologyEx(map_array, cv2.MORPH_OPEN, kernel)
    map_array = np.flipud(map_array)
    map_array = np.pad(map_array, ((0, 200), (0, 200)), 'constant', constant_values=128)
    return map_array


def main():
    map_array = prep_map('map5.pgm')
    print(f"Tamanho do mapa após processamento: {map_array.shape}")
    print(f"Valores únicos no mapa: {np.unique(map_array)}")
    print("Mapa processado:")
    plt.imshow(map_array, cmap='gray')
    plt.title("Mapa Processado")
    plt.axis('equal')
    plt.imsave('processed_map.png', map_array, cmap='gray')

    candidate = AStarPathfinder(map_array, (60, 20), (60, 120))
    candidate.run()


if __name__ == '__main__':
    main()
