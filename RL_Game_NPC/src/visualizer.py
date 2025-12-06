"""Pygame 기반 시각화"""
import pygame
import numpy as np
from typing import Dict, Optional

from .config import TileType, RoleType, VisualConfig, EnvConfig


class GameVisualizer:
    """게임 시각화 (Pygame)"""

    def __init__(self, env_config: EnvConfig, visual_config: Optional[VisualConfig] = None):
        self.env_config = env_config
        self.config = visual_config or VisualConfig()

        # Pygame 초기화
        pygame.init()

        # 화면 크기 계산
        self.screen_width = env_config.map_width * self.config.cell_size
        self.screen_height = env_config.map_height * self.config.cell_size + 100  # 정보 표시 공간

        # 화면 생성
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("5vs5 Multi-Agent Battle")

        # 폰트
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # 클럭
        self.clock = pygame.time.Clock()

        # 역할별 모양
        self.role_shapes = {
            RoleType.TANK: 'square',
            RoleType.DEALER: 'triangle',
            RoleType.HEALER: 'circle',
            RoleType.RANGER: 'diamond',
            RoleType.SUPPORT: 'pentagon',
        }

    def render(self, state: Dict) -> bool:
        """게임 상태 렌더링. False 반환 시 종료"""
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

        # 화면 클리어
        self.screen.fill((30, 30, 30))

        # 맵 렌더링
        self._render_map(state['map'])

        # 유닛 렌더링
        self._render_units(state['units'])

        # 정보 표시
        self._render_info(state)

        # 화면 업데이트
        pygame.display.flip()

        # FPS 제한
        self.clock.tick(self.config.fps)

        return True

    def _render_map(self, tile_map: np.ndarray):
        """맵 타일 렌더링"""
        for y in range(tile_map.shape[0]):
            for x in range(tile_map.shape[1]):
                tile = TileType(tile_map[y, x])
                color = self._get_tile_color(tile)

                rect = pygame.Rect(
                    x * self.config.cell_size,
                    y * self.config.cell_size,
                    self.config.cell_size,
                    self.config.cell_size
                )

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)  # 격자선

    def _get_tile_color(self, tile: TileType):
        """타일 색상 반환"""
        color_map = {
            TileType.EMPTY: self.config.color_empty,
            TileType.WALL: self.config.color_wall,
            TileType.DANGER: self.config.color_danger,
            TileType.BUFF_ATK: self.config.color_buff_atk,
            TileType.BUFF_DEF: self.config.color_buff_def,
            TileType.BUFF_HEAL: self.config.color_buff_heal,
        }
        return color_map.get(tile, self.config.color_empty)

    def _render_units(self, units: Dict):
        """유닛 렌더링"""
        for agent_id, unit_data in units.items():
            if not unit_data['is_alive']:
                continue

            x = unit_data['x']
            y = unit_data['y']
            hp = unit_data['hp']
            max_hp = unit_data['max_hp']
            team_id = unit_data['team_id']
            role = unit_data['role']

            # 유닛 위치 (픽셀)
            center_x = x * self.config.cell_size + self.config.cell_size // 2
            center_y = y * self.config.cell_size + self.config.cell_size // 2

            # 팀 색상
            color = self.config.color_team_a if team_id == 0 else self.config.color_team_b

            # 역할에 따른 모양 그리기
            self._draw_unit_shape(center_x, center_y, role, color)

            # HP 바
            self._draw_hp_bar(x, y, hp, max_hp)

    def _draw_unit_shape(self, cx: int, cy: int, role: RoleType, color):
        """역할에 따른 유닛 모양 그리기"""
        size = self.config.cell_size // 2 - 4

        if role == RoleType.TANK:
            # 사각형
            rect = pygame.Rect(cx - size, cy - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)

        elif role == RoleType.DEALER:
            # 삼각형
            points = [
                (cx, cy - size),
                (cx - size, cy + size),
                (cx + size, cy + size)
            ]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)

        elif role == RoleType.HEALER:
            # 원
            pygame.draw.circle(self.screen, color, (cx, cy), size)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), size, 2)
            # 십자가 표시
            pygame.draw.line(self.screen, (255, 255, 255),
                           (cx - size//2, cy), (cx + size//2, cy), 2)
            pygame.draw.line(self.screen, (255, 255, 255),
                           (cx, cy - size//2), (cx, cy + size//2), 2)

        elif role == RoleType.RANGER:
            # 다이아몬드
            points = [
                (cx, cy - size),
                (cx + size, cy),
                (cx, cy + size),
                (cx - size, cy)
            ]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)

        elif role == RoleType.SUPPORT:
            # 오각형
            import math
            points = []
            for i in range(5):
                angle = math.radians(90 + i * 72)
                px = cx + int(size * math.cos(angle))
                py = cy - int(size * math.sin(angle))
                points.append((px, py))
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)

    def _draw_hp_bar(self, x: int, y: int, hp: int, max_hp: int):
        """HP 바 그리기"""
        bar_width = self.config.cell_size - 8
        bar_height = 4
        bar_x = x * self.config.cell_size + 4
        bar_y = y * self.config.cell_size + 2

        # 배경
        pygame.draw.rect(self.screen, self.config.color_hp_bg,
                        (bar_x, bar_y, bar_width, bar_height))

        # HP
        hp_width = int(bar_width * hp / max_hp)
        hp_color = (0, 255, 0) if hp / max_hp > 0.5 else (255, 255, 0) if hp / max_hp > 0.25 else (255, 0, 0)
        pygame.draw.rect(self.screen, hp_color,
                        (bar_x, bar_y, hp_width, bar_height))

    def _render_info(self, state: Dict):
        """게임 정보 표시"""
        info_y = self.env_config.map_height * self.config.cell_size + 10

        # 턴 정보
        step_text = self.font.render(f"Step: {state['step']}", True, (255, 255, 255))
        self.screen.blit(step_text, (10, info_y))

        # 팀 정보
        team_a_alive = sum(1 for aid, u in state['units'].items()
                          if u['team_id'] == 0 and u['is_alive'])
        team_b_alive = sum(1 for aid, u in state['units'].items()
                          if u['team_id'] == 1 and u['is_alive'])

        team_a_text = self.font.render(f"Team A: {team_a_alive}/5", True, self.config.color_team_a)
        team_b_text = self.font.render(f"Team B: {team_b_alive}/5", True, self.config.color_team_b)
        self.screen.blit(team_a_text, (200, info_y))
        self.screen.blit(team_b_text, (350, info_y))

        # 승자 표시
        if state['winner'] is not None:
            winner = "Team A" if state['winner'] == 0 else "Team B"
            winner_text = self.font.render(f"Winner: {winner}!", True, (255, 255, 0))
            self.screen.blit(winner_text, (500, info_y))

        # 범례
        legend_y = info_y + 30
        legend_text = self.small_font.render("Tank:Square  Dealer:Triangle  Healer:Circle  Ranger:Diamond  Support:Pentagon",
                                             True, (200, 200, 200))
        self.screen.blit(legend_text, (10, legend_y))

        # 조작 안내
        help_text = self.small_font.render("Press ESC to quit", True, (150, 150, 150))
        self.screen.blit(help_text, (10, legend_y + 20))

    def close(self):
        """시각화 종료"""
        pygame.quit()

    def wait_for_key(self):
        """키 입력 대기"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    else:
                        waiting = False
        return True