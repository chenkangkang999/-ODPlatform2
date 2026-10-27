# 鱼类中文展示名

本表仅服务于 Web 类别卡片和中文搜索。训练配置、类别编号、标注文件和模型输出仍使用原始类别名，顺序不变。中文没有统一可靠名称时采用当地名音译；不把展示名视为新增物种鉴定结论。

依据 [BengalDeltaFish 原始论文表 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC12221697/) 核对当地名所指类别。表中“通俗译名”是对论文物种名称或英文俗名的中文表达，“音译”是便于中文界面阅读的展示名称，不声称为标准中文学名。

| 原始类别 | 中文展示名 | 处理方式 |
| --- | --- | --- |
| Batashi | 巴塔希鱼 | 当地名音译 |
| Bele | 贝勒鱼 | 当地名音译 |
| Chanda | 印度玻璃鱼 | 论文英文俗名的通俗译名 |
| Chapila | 印度河鲱 | 论文英文俗名的通俗译名 |
| Chela | 切拉鱼 | 当地名音译 |
| Foli | 弓背鱼 | FishBase 中文俗名 |
| Ilish | 云鲥 | FishBase 中文俗名 |
| Kajuli | 卡朱利鱼 | 当地名音译 |
| Kayakanta | 卡亚坎塔鱼 | 当地名音译 |
| Koi | 攀鲈 | FishBase 中文俗名 |
| Magur | 斑纹胡鲶 | FishBase 中文俗名 |
| Mola | 莫拉鱼 | 当地名音译 |
| Mrigel | 姆里格尔鱼 | 当地名音译 |
| Pabda | 帕布达鱼 | 当地名音译 |
| Pangas | 潘加斯鱼 | 当地名音译 |
| Puti | 普蒂鱼 | 当地名音译 |
| Rui | 罗胡鱼 | 当地名/通行名称音译 |
| Rupchada | 中国鲳 | 论文物种名称的通俗译名 |
| Shol | 条纹鳢 | 论文英文俗名的通俗译名 |
| Sorputi | 橄榄鲃 | 论文英文俗名的通俗译名 |
| Taki | 斑点鳢 | 论文英文俗名的通俗译名 |
| Tarabaim | 塔拉拜姆鱼 | 当地名音译 |
| Telapia | 莫桑比克罗非鱼 | 对论文所列 Mozambique tilapia 的中文表达 |
| Tengra | 滕格拉鱼 | 当地名音译 |
| Veda | 印度叶鱼 | 论文英文俗名的通俗译名 |
| Vetki | 尖吻鲈 | FishBase 中文俗名 |
| ayer | 艾尔鱼 | 当地名音译 |
| boyal | 波亚尔鱼 | 当地名音译 |
| chitol | 奇托尔鱼 | 当地名音译，避免论文表中学名重复带来的误配 |
| kalbaus | 卡尔巴斯鱼 | 当地名音译 |
| katol | 卡特拉鱼 | 当地名/通行名称音译 |
| kholshe | 条纹丝足鱼 | 论文英文俗名的通俗译名 |
| shing | 辛格鱼 | 当地名音译 |

特别核对：这里的 Koi 是论文所列的 Anabas testudineus，故显示“攀鲈”；它不是锦鲤。原论文对 chitol 和 Foli 重复列出同一学名，因此 chitol 保留音译，避免在展示层擅自修订数据类别。

FishBase 核对页面：

- [Foli / 弓背鱼](https://fishbase.se/ComNames/CommonNameSummary.php?autoctr=279291)
- [Ilish / 云鲥](https://www.fishbase.se/ComNames/CommonNameSummary.php?autoctr=288818)
- [Koi / 攀鲈](https://fishbase.se/ComNames/CommonNamesList.php?GenusName=Anabas&ID=495&SpeciesName=testudineus&StockCode=511)
- [Magur / 斑纹胡鲶](https://www.fishbase.se/Country/CountryChecklist.php?c_code=156&showAll=yes&vhabitat=fresh)
- [Telapia 对应物种](https://www.fishbase.se/Fieldguide/FieldGuideSummary.php?c_code=156&genusname=Oreochromis&speciesname=mossambicus)
- [Vetki / 尖吻鲈](https://www.fishbase.se/FieldGuide/FieldGuideSummary.php?c_code=156&genusname=Lates&speciesname=calcarifer)

实现位于 `apps/web-frontend/src/fish-labels.js`；映射忽略大小写和首尾空格，未知类别回退显示“鱼类”。
