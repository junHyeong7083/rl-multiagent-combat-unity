"""맵(Map) 시스템 구현"""
import numpy as np
from typing import List, Tuple, Optional
import random

from .config import TileType, EnvConfig


class GameMap:
    """2D 격자 기반 게임 맵"""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.width = config.map_width
        self.height = config.map_height

        # 타일 맵 (2D 배열)
        self.tiles: np.ndarray = np.zeros((self.height, self.width), dtype=np.int32)

        # 유닛 위치 추적 (좌표 -> 유닛 ID)
        self.unit_positions: dict = {}

    def generate(self, seed: Optional[int] = None):
        """맵 랜덤 생성"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 모든 타일을 평지로 초기화
        self.tiles.fill(TileType.EMPTY)

        # 벽 생성 (가장자리와 랜덤 위치)
        self._generate_walls()

        # 위험 타일 생성
        self._generate_danger_tiles()

        # 버프 타일 생성
        self._generate_buff_tiles()

    def _generate_walls(self):
        """벽 생성"""
        # 맵 가장자리는 벽으로 둘러싸기 (선택사항, 주석 처리)
        # self.tiles[0, :] = TileType.WALL
        # self.tiles[-1, :] = TileType.WALL
        # self.tiles[:, 0] = TileType.WALL
        # self.tiles[:, -1] = TileType.WALL

        # 중앙에 약간의 장애물 배치
        num_walls = int(self.width * self.height * self.config.wall_density)

        for _ in range(num_walls):
            x = random.randint(2, self.width - 3)
            y = random.randint(2, self.height - 3)

            # 스폰 지역은 벽 생성 금지 (왼쪽/오른쪽 가장자리 근처)
            if x < 4 or x > self.width - 5:
                continue

            self.tiles[y, x] = TileType.WALL

    def _generate_danger_tiles(self):
        """위험 타일 생성"""
        num_danger = int(self.width * self.height * self.config.danger_density)

        for _ in range(num_danger):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)

            if self.tiles[y, x] == TileType.EMPTY:
                self.tiles[y, x] = TileType.DANGER

    def _generate_buff_tiles(self):
        """버프 타일 생성"""
        num_buffs = int(self.width * self.height * self.config.buff_density)

        buff_types = [TileType.BUFF_ATK, TileType.BUFF_DEF, TileType.BUFF_HEAL]

        for _ in range(num_buffs):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)

            if self.tiles[y, x] == TileType.EMPTY:
                self.tiles[y, x] = random.choice(buff_types)

    def is_valid_position(self, x: int, y: int) -> bool:
        """유효한 위치인지 확인"""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        """이동 가능한 위치인지 확인"""
        if not self.is_valid_position(x, y):
            return False
        return self.tiles[y, x] != TileType.WALL

    def is_occupied(self, x: int, y: int) -> bool:
        """유닛이 있는 위치인지 확인"""
        return (x, y) in self.unit_positions

    def can_move_to(self, x: int, y: int) -> bool:
        """이동 가능 여부 (벽과 유닛 모두 체크)"""
        return self.is_walkable(x, y) and not self.is_occupied(x, y)

    def get_tile(self, x: int, y: int) -> TileType:
        """특정 위치의 타일 타입 반환"""
        if not self.is_valid_position(x, y):
            return TileType.WALL  # 맵 밖은 벽 취급
        return TileType(self.tiles[y, x])

    def place_unit(self, unit_id: int, x: int, y: int):
        """유닛을 맵에 배치"""
        self.unit_positions[(x, y)] = unit_id

    def remove_unit(self, x: int, y: int):
        """유닛을 맵에서 제거"""
        if (x, y) in self.unit_positions:
            del self.unit_positions[(x, y)]

    def move_unit(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]):
        """유닛 위치 이동"""
        if old_pos in self.unit_positions:
            unit_id = self.unit_positions[old_pos]
            del self.unit_positions[old_pos]
            self.unit_positions[new_pos] = unit_id

    def get_spawn_positions(self, team_id: int, num_units: int) -> List[Tuple[int, int]]:
        """팀별 스폰 위치 생성"""
        positions = []

        if team_id == 0:  # Team A: 왼쪽
            base_x = 2
        else:  # Team B: 오른쪽
            base_x = self.width - 3

        # 세로로 분산 배치
        spacing = self.height // (num_units + 1)

        for i in range(num_units):
            y = spacing * (i + 1)
            x = base_x

            # 해당 위치가 사용 불가능하면 주변 탐색
            if not self.can_move_to(x, y):
                found = False
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if self.can_move_to(nx, ny):
                            x, y = nx, ny
                            found = True
                            break
                    if found:
                        break

            positions.append((x, y))

        return positions

    def get_local_observation(self, x: int, y: int, radius: int) -> np.ndarray:
        """특정 위치 주변의 타일 정보 반환 (관찰용)"""
        size = 2 * radius + 1
        obs = np.zeros((size, size), dtype=np.int32)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                obs_y, obs_x = dy + radius, dx + radius

                if self.is_valid_position(nx, ny):
                    obs[obs_y, obs_x] = self.tiles[ny, nx]
                else:
                    obs[obs_y, obs_x] = TileType.WALL  # 맵 밖은 벽

        return obs

    def update_danger_tiles(self, step: int):
        """위험 타일 업데이트 (시간에 따라 변화)"""
        # 매 10턴마다 위험 타일이 변화
        if step % 10 == 0:
            # 기존 위험 타일 일부 제거
            danger_positions = np.where(self.tiles == TileType.DANGER)
            for y, x in zip(danger_positions[0], danger_positions[1]):
                if random.random() < 0.3:  # 30% 확률로 제거
                    self.tiles[y, x] = TileType.EMPTY

            # 새로운 위험 타일 생성
            num_new = random.randint(1, 3)
            for _ in range(num_new):
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if self.tiles[y, x] == TileType.EMPTY:
                    self.tiles[y, x] = TileType.DANGER

    def get_state_matrix(self) -> np.ndarray:
        """전체 맵 상태를 행렬로 반환"""
        return self.tiles.copy()

    def reset(self):
        """맵 리셋"""
        self.unit_positions.clear()
        self.generate()

    def __repr__(self):
        return f"GameMap({self.width}x{self.height})"