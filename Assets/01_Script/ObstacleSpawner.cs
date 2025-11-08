using UnityEngine;

public class ObstacleSpawner : MonoBehaviour
{
    [Header("Map")]
    public Vector2Int gridSize = new Vector2Int(64, 64);
    public float cellSize = 1f;


    [Header("Noise")]
    public float noiseScale = 0.12f;
    public float threshold = 0.48f; // 이 값 이하만 장애물 생성
    public int seed = 0;

    [Header("ObstaclePrefab")]
    public GameObject obstaclePrefab;

    [Header("Rules")]
    public int safeRadius = 4; // 스폰/목표 구역 보호 반경(격자 단위)
    public Vector2Int spawnA = new Vector2Int(4, 4);
    public Vector2Int spawnB = new Vector2Int(60, 60);

    Transform container;

    public void Generate(int _episodeSeed)
    {
        seed = _episodeSeed;

        if(container != null) Destroy(container);
        container = new GameObject("Obstacle").transform;
        container.SetParent(transform,false);

        float offX = seed * 0.12345f; // 시드 오프셋
        float offY = seed * 0.54321f;

        for (int y = 0; y < gridSize.y; y++)
        {
            for (int x = 0; x < gridSize.x; x++)
            {
                // 스폰 보호
                if (Vector2Int.Distance(new Vector2Int(x, y), spawnA) < safeRadius) continue;
                if (Vector2Int.Distance(new Vector2Int(x, y), spawnB) < safeRadius) continue;

                float nx = (x + offX) * noiseScale;
                float ny = (y + offY) * noiseScale;
                float v = Mathf.PerlinNoise(nx, ny);
                if (v <= threshold)
                {
                    Vector3 pos = new Vector3(x * cellSize, 0f, y * cellSize);
                    var o = Instantiate(obstaclePrefab, pos, Quaternion.identity, container);
                    o.layer = LayerMask.NameToLayer("Obstacle"); 
                }
            }
        }
         
    }
}
