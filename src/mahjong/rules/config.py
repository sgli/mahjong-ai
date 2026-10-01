"""Rules configuration (Phase: rules configurisation, step 1).

A single :class:`RulesConfig` records the rule-set choices that used to be
hard-coded Tenhou assumptions, so they can be traced in metadata and switched
without touching the environment logic.

Only ``atamahane`` (head-bump vs multi-ron) is wired end-to-end in step 1.
The other fields are wired mechanically (on/off switches) or are documented
placeholders pending step 2.

Sources for Majsoul values (verified via web search):
- ``atamahane=False`` (multi-ron / no head bump):
  https://wapbaike.baidu.com/item/雀魂麻将/60778444 — "无头跳抢和，有多家和牌".
- ``agari_yame=True``:
  https://moegirl.uk/卡维(雀魂麻将) — 南四局结束时庄家为一位超过30000点且满足连庄时不再连庄.
- ``west_round=True``:
  https://mobile.moegirl.org.cn/雀魂麻将 — 东南战西入，无玩家超过30000点时加时赛.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RulesConfig:
    name: str
    rules_id: str
    atamahane: bool = True  # 头跳（True=最近座位胜，False=多家和牌）
    four_wind_abort: bool = True  # 四风连打
    four_kan_abort: bool = True  # 四杠散了
    four_riichi_abort: bool = True  # 四家立直
    nine_terminals_abort: bool = True  # 九种九牌
    agari_yame: bool = True  # 和了止め（all last top 和牌即终局）
    west_round: bool = True  # 西入
    red_fives: int = 3  # 赤宝牌张数（0=无，3=五万/五饼/五索各 1）
    nagashi_mangan: bool = True  # 流局满贯
    double_yakuman: bool = False  # 双倍役满（step3 待核实/接线）
    kiriage_mangan: bool = False  # 切上满贯（step2 接线）

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RulesConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


#: Tenhou rules — the default (current behaviour).
TENHOU_RULES = RulesConfig(
    name="tenhou",
    rules_id="tenhou-v1",
    atamahane=True,
    four_wind_abort=True,
    four_kan_abort=True,
    four_riichi_abort=True,
    nine_terminals_abort=True,
    agari_yame=True,
    west_round=True,
    red_fives=3,
    nagashi_mangan=True,  # 用户确认：天凤/雀魂均有流局满贯；mjai 数据不编码流局满贯结算，验证器跳过 nagashi 重建（见 validate_rules.py）
    double_yakuman=False,  # step3 待核实（当前按单倍役满）
    kiriage_mangan=False,  # 天凤不使用切上满贯（validate_rules 验证 4番30符=11600 非满贯）
)

#: Majsoul rules — multi-ron wired (step 1); kiriage_mangan wired (step 2);
#: 流局种类/切上满贯/赤宝牌/流局满贯 由用户确认（见下）。
MAJSOUL_RULES = RulesConfig(
    name="majsoul",
    rules_id="majsoul-v1",
    atamahane=False,  # 来源: 百度百科 雀魂麻将 "无头跳抢和，有多家和牌"
    four_wind_abort=True,  # 用户确认：同天凤
    four_kan_abort=True,  # 用户确认：同天凤
    four_riichi_abort=True,  # 用户确认：同天凤
    nine_terminals_abort=True,  # 用户确认：同天凤
    agari_yame=True,  # 来源: moegirl 卡维(雀魂麻将)
    west_round=True,  # 来源: moegirl 雀魂麻将 "东南战西入"
    red_fives=3,  # 用户确认：5条/5万/5筒 各 1 张（=3）
    nagashi_mangan=True,  # 用户确认：雀魂有流局满贯
    double_yakuman=False,  # step3 待核实（暂同天凤）
    kiriage_mangan=False,  # 用户确认：雀魂无切上满贯
)


__all__ = ["MAJSOUL_RULES", "RulesConfig", "TENHOU_RULES"]
