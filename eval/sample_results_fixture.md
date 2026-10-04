# Sample eval output (SYNTHETIC fixture data, not real Met results)

300 query objects from a pool of 1000; k = 10; model `fake/color-thumb+hashed-bow`; generated 2026-10-04T10:07:17+00:00.

| Attribute | Model | Random | Lift | Queries with value |
|---|---|---|---|---|
| department | 87.2% | 24.8% | 3.5x | 300 |
| culture | 87.2% | 19.1% | 4.6x | 300 |
| classification | 88.4% | 21.4% | 4.1x | 300 |
| period | 87.1% | 15.6% | 5.6x | 171 |
| any | 88.5% | 32.2% | 2.7x | 300 |

`any` = neighbour shares at least one of department, culture, classification.

## By department (model)

| Department | department | culture | classification | period |
|---|---|---|---|---|
| Asian Art | 85.6% | 85.6% | 88.7% | 85.4% |
| Drawings and Prints | 94.0% | 94.0% | 94.0% | n/a |
| Egyptian Art | 88.8% | 88.8% | 88.8% | 88.8% |
| European Paintings | 78.6% | 78.6% | 80.2% | n/a |

## Worst queries

- **Woman Reading No. 3757** (3757; italian, paintings): 10.0% of neighbours share anything. Neighbours: Woman Reading No. 1736 (prints); Woman Reading No. 3694 (paintings); Woman Reading No. 1561 (prints); Woman Reading No. 1035 (prints); Woman Reading No. 1779 (prints)
- **Landscape No. 3294** (3294; egyptian, sculpture): 20.0% of neighbours share anything. Neighbours: Landscape No. 2113 (drawings); Landscape No. 3241 (sculpture); Landscape No. 2315 (drawings); Bird No. 1999 (drawings); Bird No. 2461 (drawings)
- **Flowers No. 2185** (2185; dutch, drawings): 20.0% of neighbours share anything. Neighbours: Flowers No. 3376 (sculpture); Flowers No. 2057 (drawings); Flowers No. 3360 (sculpture); Flowers No. 2618 (sculpture); Flowers No. 3373 (sculpture)
- **Landscape No. 2948** (2948; egyptian, sculpture): 20.0% of neighbours share anything. Neighbours: Landscape No. 3114 (sculpture); Landscape No. 1799 (drawings); Landscape No. 1983 (drawings); Landscape No. 2212 (drawings); Landscape No. 2315 (drawings)
- **River No. 2667** (2667; egyptian, sculpture): 20.0% of neighbours share anything. Neighbours: River No. 3517 (paintings); River No. 3552 (paintings); River No. 4159 (paintings); River No. 3074 (sculpture); River No. 4029 (paintings)
- **River No. 4128** (4128; italian, paintings): 30.0% of neighbours share anything. Neighbours: River No. 1011 (prints); River No. 1030 (prints); River No. 3691 (paintings); River No. 1285 (prints); River No. 1232 (prints)
- **Mountain No. 3904** (3904; italian, paintings): 30.0% of neighbours share anything. Neighbours: Mountain No. 3264 (sculpture); Mountain No. 3506 (paintings); Mountain No. 2944 (sculpture); River No. 4029 (paintings); River No. 3724 (paintings)
- **Flowers No. 3429** (3429; italian, paintings): 30.0% of neighbours share anything. Neighbours: Flowers No. 3974 (paintings); Flowers No. 1373 (prints); Flowers No. 1765 (prints); Flowers No. 3859 (paintings); Flowers No. 1048 (prints)
