// Display labels only. Keep dataset names and class IDs unchanged.
// Naming sources and transliterations: docs/FISH_CHINESE_LABELS.md.
const chineseNames = Object.freeze({
  batashi: '巴塔希鱼',
  bele: '贝勒鱼',
  chanda: '印度玻璃鱼',
  chapila: '印度河鲱',
  chela: '切拉鱼',
  foli: '弓背鱼',
  ilish: '云鲥',
  kajuli: '卡朱利鱼',
  kayakanta: '卡亚坎塔鱼',
  koi: '攀鲈',
  magur: '斑纹胡鲶',
  mola: '莫拉鱼',
  mrigel: '姆里格尔鱼',
  pabda: '帕布达鱼',
  pangas: '潘加斯鱼',
  puti: '普蒂鱼',
  rui: '罗胡鱼',
  rupchada: '中国鲳',
  shol: '条纹鳢',
  sorputi: '橄榄鲃',
  taki: '斑点鳢',
  tarabaim: '塔拉拜姆鱼',
  telapia: '莫桑比克罗非鱼',
  tengra: '滕格拉鱼',
  veda: '印度叶鱼',
  vetki: '尖吻鲈',
  ayer: '艾尔鱼',
  boyal: '波亚尔鱼',
  chitol: '奇托尔鱼',
  kalbaus: '卡尔巴斯鱼',
  katol: '卡特拉鱼',
  kholshe: '条纹丝足鱼',
  shing: '辛格鱼',
})

export function fishChineseName(name) {
  return chineseNames[String(name ?? '').trim().toLowerCase()] ?? '鱼类'
}
