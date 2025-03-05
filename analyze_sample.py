import re
from typing import Dict, List, Set
from collections import defaultdict

def analyze_sample(text: str) -> Dict:
    """
    分析样本文本的各种特征
    
    Args:
        text: 输入的样本文本
        
    Returns:
        Dict: 包含分析结果的字典
    """
    stats = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "total_tokens": 0,  # 添加总token数统计
        "special_tokens": {
            "user": [],
            "item": [],
            "category": []
        },
        "normal_tokens": [],  # 添加普通token列表
        "category_hierarchy": [],
        "rating": None,
        "review_count": None
    }
    
    # 提取所有token
    tokens = text.split()
    for token in tokens:
        if token.startswith("[user_"):
            stats["special_tokens"]["user"].append(token)
        elif token.startswith("[item_"):
            stats["special_tokens"]["item"].append(token)
        elif token.startswith("[category_"):
            stats["special_tokens"]["category"].append(token)
        else:
            # 将标点符号作为单独的token
            words = re.findall(r'\w+|[^\w\s]', token)
            stats["normal_tokens"].extend(words)
    
    # 计算总token数
    stats["total_tokens"] = (len(stats["special_tokens"]["user"]) + 
                           len(stats["special_tokens"]["item"]) + 
                           len(stats["special_tokens"]["category"]) + 
                           len(stats["normal_tokens"]))
            
    # 提取类别层级
    category_match = re.search(r'category hierarchy: (.*?),', text)
    if category_match:
        hierarchy = category_match.group(1).strip()
        stats["category_hierarchy"] = [cat.strip() for cat in hierarchy.split(">")]
        
    # 提取评分和评论数
    rating_match = re.search(r'(\d+\.\d+) average rating based on (\d+) reviews', text)
    if rating_match:
        stats["rating"] = float(rating_match.group(1))
        stats["review_count"] = int(rating_match.group(2))
        
    return stats

def print_analysis(stats: Dict):
    """
    打印分析结果
    
    Args:
        stats: analyze_sample函数返回的统计信息
    """
    print("=" * 50)
    print("样本分析结果:")
    print("=" * 50)
    
    print(f"\n1. 基本统计:")
    print(f"- 文本总长度: {stats['text_length']} 字符")
    print(f"- 单词总数: {stats['word_count']} 个")
    print(f"- 总token数: {stats['total_tokens']} 个")
    print(f"  - 特殊token: {sum(len(tokens) for tokens in stats['special_tokens'].values())} 个")
    print(f"  - 普通token: {len(stats['normal_tokens'])} 个")
    
    print(f"\n2. 特殊Token统计:")
    print(f"- 用户token数量: {len(stats['special_tokens']['user'])} 个")
    print(f"- 物品token数量: {len(stats['special_tokens']['item'])} 个")
    print(f"- 类别token数量: {len(stats['special_tokens']['category'])} 个")
    
    print(f"\n3. 类别层级结构:")
    for i, category in enumerate(stats["category_hierarchy"], 1):
        print(f"  第{i}层: {category}")
        
    print(f"\n4. 评分信息:")
    print(f"- 平均评分: {stats['rating']}")
    print(f"- 评论数量: {stats['review_count']}")
    
    print(f"\n5. 详细Token列表:")
    print("\n物品Token:")
    for item in stats['special_tokens']['item']:
        print(f"  {item}")
        
    print("\n类别Token:")
    for cat in stats['special_tokens']['category']:
        print(f"  {cat}")
        
    print(f"\n用户Token数量过多，仅显示前5个:")
    for user in stats['special_tokens']['user'][:5]:
        print(f"  {user}")
    print(f"  ... 等共{len(stats['special_tokens']['user'])}个用户")
    
    print("\n" + "=" * 50)

def main():
    # 示例文本
    sample_text = '[item_B002MZ8BK2] belongs to category hierarchy: [category_Beauty] > [category_Beauty|Tools_Accessories] > [category_Beauty|Nail_Tools] > [category_Beauty|Nail_Art_Equipment] , with 4.4 average rating based on 264 reviews. It has been purchased by the following users: [user_A3A5OJ5BGWJYF4] [user_A34C4YYZP4IL2G] [user_A3HKGKK1ZHA7AG] [user_A3PPY2UX9K8B0V] [user_A28828YVMWGDFG] [user_A36Z2WORF8Y6WX] [user_A1APCGMUVYINAJ] [user_A115HT86H2I79N] [user_AZRPMQ8A4H0W4] [user_A35BNLIJ3LXGMH] [user_A3K5QN33V0L457] [user_AU49TV5S64GT8] [user_A3DKP8M0GSP8UK] [user_A306GI4J03UI88] [user_A37IIX7OLELUMT] [user_AKMBOKJQBLI1U] [user_A81DWIYDPB9XW] [user_A1CF9MFGAI6IBO] [user_A1S85HGM10I5VY] [user_A1PPH3OM99MH7Y] [user_A38ZGA4I90MM8Q] [user_A2TS77S3W0U36U] [user_A33X2WO55U81KR] [user_A9SWXSVCHJS6L] [user_AEQWORECFS8UH] [user_A3UQ5RCJLA6RV2] [user_A3SKKX50FWXO45] [user_A33G2GBQGONZK] [user_ANGJ5BUZ1H1DP] [user_A20H2Q6KA5D4B3] [user_A3BKP8K7NS1BRA] [user_A3S7ORXU4KYABQ] [user_A2LEGWIA9NDNQB] [user_A2M8T1C3YP19NQ] [user_A1C7U2JM39BS28] [user_AI1YKL18DPF03] [user_A3AQ6I1N1VA8VJ] [user_A27LI8T4Y6AG2K] [user_A3B1OIVLLIWKFY] [user_A2XMJPUHYRY56K] [user_A1R6RMK6L45SJP] [user_A2BR1JRTBH5QLQ] [user_AIUR2Z8HQMNSS] [user_A1PUN0QQ2Q6Y0N] [user_A181T1U7WRODR9] [user_AKY2MJ46YSAP0] [user_A2VOMWM3O95P7O] [user_A2UNBBMNPBM7OJ] [user_A2R1VMGSXJ272L] [user_A1BN17AK4KO91H] [user_A31QE2DFZOA2HJ] [user_AL39XR7EYLOBD] [user_A2K0ZW0UPPKUEX] [user_A1P7TTDNC9M4DD] [user_A2JLVMIGLOH5SV] [user_A3IGQMALGUJJUV] [user_A2V7ET4KLETE1M] [user_A2OL6MVLQRMH3U] [user_A01198201H0E3GHV2Z17I] [user_A2KG5YIV44VGYU] [user_A4OU7GRF8W0Y8] [user_A3EIDGJ92SXMQK] [user_A35O3IFYBBM4Q1] [user_AD29UXW42GRA4] [user_A38MTIH211HAGE] [user_ASW3I4V90PWH4] [user_ARAGORUBUH8XT] [user_A3CSHG6B6933CO] [user_A2WLQEY91NZMU0] [user_A3CEMAMNAU6SLZ] [user_A3DKX4KQIW9RUT] [user_A3SCU8S8MC6S4Q] [user_A1XWO6BJJF7PP] [user_A39RDQEBQV6BB6] [user_A15V4TJBVYLK94] [user_A23RK31HI945OU] [user_A2BUUYQ2DDAKUE] [user_A14RZOHFA6TCXY] [user_A37OLD09U3W17Y] [user_A3GUOT46FKC58M] [user_A2BVF1JUKTDDSW] [user_A25TVH9W5A6IHP] [user_AH5NJQM5TC7SW] [user_AMFJSMJJ8FCV6] [user_A2PGDUYXZFAPT5] [user_A36AXK5GNMKFLX] [user_A2Z3RL11ABBFIS] [user_A1W13I62KL7UVX] [user_A1SUZPMEHYUH1M] [user_A3SJ6UTUVWFS7K] [user_A1COZ8USA4RICX] [user_A3QQPTTCHWU2RK] [user_AT8K2X9IQS407] [user_A1WZK1FN3S355V] [user_A2P2O3I4C7OJC5] [user_A1TNRVE3TXZIUF] [user_A18YQ1GK642VS4] [user_A2I02GDZCKPPZQ] [user_ACVB8KR85ZT3J] [user_A3D9V8IXRN4PIT] [user_A2MGYVPEUJHX1V] [user_A1HW8AKTZ46Y3I] [user_A8V9AM4WK5AFU] [user_AOFMA2QQUTN2] [user_A2NIIB3VA49IG0] [user_A1PSSB1PTAFOS5] [user_A103979529MRJY0U56QI4] [user_A2I7MP5GCRNNF0] [user_A30RVE6QWTD584] [user_A3GHJ7MQ57PJFV] [user_ANTWLVOKZFISX] [user_AM4TUW0HWNBEV] [user_A1OAFXHHRB8BX8] [user_A2G9QJBQ5OZRHD] [user_A2LH3CBOXMPWJO] [user_A7QZ3NG2GL2CL] [user_A36G31SDOQTNGM] [user_A25K93BZ8GX9ZC] [user_A1M4KL2EMZVSJU] [user_A3AHRHD38SY22E] [user_A3CSRH0F0WQHWY] [user_A108S8GV9IJJ96] [user_A1IJ985V27NESS] [user_A2O952LCMLPDX8] [user_ASBUN4IBPXXIC] [user_A1NYMOSVAAV2KY] [user_AP45U0HTF51HR] [user_A3HI2YW6CGRYPL] [user_A2AK7GOCU0JZ4L] [user_AJO6T9BSZPNCF] [user_A2FXHRM0QNU0QL] [user_A27A64XRTQKJW2] [user_AC2XZ2EOOTIUO] [user_A2QRUJQ4V9VD9C] [user_A1429179XSHSTD] [user_AX9RDX5JCSPI4] [user_A1CIMOIAZUR7CP] [user_A1HXB7J6HQMAM5] [user_A2ABYMQKF6TLR9] [user_A1O66ZGPF1EYAR] [user_A1MUFYWR8YSS74] [user_A3N3ZTC5LDWDWD] [user_A2C3A5EFYIIKOM] [user_A1L9HCF991W6W9] [user_A3SUH4HE6Y4VLW] [user_A2V4I78RMTHLW6] [user_A3UXZ1AZUS6E02] [user_A1I6ICEIWYF5H0] [user_A2TRO9OD7QWO0T] [user_A3W92J2NDDNC4] [user_A1D2Q0ZNMIGKD5] [user_A205TC3P6DLYT0] [user_A34BWSH3XD1IMO] [user_A2KHBNINW82BCO] [user_A2EF1UPULT0VAX] [user_A1M0OSMWJWL33N] [user_A13K940O9R7HSK] [user_A5Z2YW5LIW5W7] [user_A3TUV4NXPAQ1Y6] [user_A3LIN7EQRJ03SX] [user_A2VQB3Q40Q14AZ] [user_A25LSPT16HKTIE] [user_A22GDD8RRV6D89] [user_A3U1U98WH5ZD8F] [user_APAAR4GZK32T4] [user_A36FIOWY2S2S8J] [user_A4W8OI7G8OTHE] [user_AHJJ9UFSUHIF6] [user_A18U741AFDT54Y] [user_A2L7HKSC01X1G7] [user_AAZ39X5Q8SAOU] [user_ANMJ6RVCV4KXI] [user_A3NV9TGNKHLTZH] [user_AXMFZBD96CA3M] [user_A2BEFIJOBPFVPA] [user_A1BBNDXVMTRQPZ] [user_A13KAOEXRR8OQC] [user_ANXPB7ON5N4D5] [user_A1NRNRWMVN04I5] [user_A2QVYND4OO835K] [user_A1BO9AIZNUYTK8] [user_A3TMBYDFP1B9DM] [user_A2B940WUFUXMTQ] [user_A22QOZZLLIMW6I] [user_A1HG6G029EQVZX] [user_A2M09NBM1IIJE9] [user_A3NKBCNIXTL2CM] [user_A1XSYYLTSL1LEC] [user_A27G28RV3EWDSF] [user_A16SBVY0SF5OF1] [user_A16EX7FYCNQNJD] [user_A21IYMZ6LAPK8D] [user_A1EBL4FNVLDE6W] [user_AMAEN1XO8STCN] [user_A1YRLG0G615QH] [user_A24FQNZ2ZCP9UH] [user_A113VGLEOCAE9G] [user_A1TBOZXU3ZFCZO] [user_A1TLMKTISOXX3Y] [user_A1XYQ7GX8QREJN] [user_A1BS22O15EOQ9S] [user_AW1Z4YECR268E] [user_A2C7HS98O3MDT0] [user_A1QQVLG6F6CH2N] [user_A1OPRLUUQRCDAB] [user_A3JZJVHQFDEQXO] [user_A2J7DA0G60DEPJ] [user_A1121T1I50825Z] [user_A37YEEC9CL7ETX] [user_AOFFWMV2JISWR] [user_A1VNDWJ9PGZBHK] [user_AV3D1L0MWSW7K] [user_A1OKHKMDD9EGJT] [user_AX47EVO5R71HI] [user_A3E9Z9JCES29F] [user_A37XHG5LLONQH2] [user_A1QMHEZD6SU24H] [user_A83VUXVIDWITV] [user_A1V106YS1KRAMV] [user_ALYXQYKO4F00Z] [user_A1WU0NHEV9E5K1] [user_A2T414T54X8A12] [user_A3KIBLDPW1D6P9] [user_A3PMVIRPB2U82Y] [user_AEE9GBSOWORSI] [user_AX71GBUUW2PPT] [user_A1L3BUW35KO49C] [user_A2120VDBOIOVWH] [user_A324KU3G9I4D5V] [user_A1KPLM4USSPHHB] [user_APG8YOWMAHRPG] [user_A59C1X49WAXFH] [user_A2HMP5LYU0BF6B] [user_A1OLS38AAEFH37] [user_A2Q4OHFW1PIOOU] [user_AJKN1OQ368JGH] [user_A1GBZ53N4JYC5G] [user_A3I6403ZZ08SG4] [user_A304SXKQUCT02I] [user_A22V040XW6MGZL] [user_A29UZ3E0QILZOC] [user_A1NH1FRM2SZL7W] [user_A1YSXBJFTLNSXM] [user_A2WCMHMNZ35FEO] [user_A2ND7WY43CKE3K] [user_APZEJTY5U5ILS] [user_A19JTNN6JMIHSR] [user_A261X789JXBS20] [user_A2AXBMMKOXUDGV] [user_AIB5B9WTNVEKC] [user_A3IKA9SP1N0BSK] [user_A3LGFZ5P8U3TZ1] [user_A3PUVT9HWURYNM] [user_ASMK582FKPNDA] [user_A1VN9S26YLG3BI] [user_AXZ746UBDPNO6] [user_A1NGY9657YA3CQ] [user_AN6N0O00PMYHM] [user_A37A9AW2IEJY5T] [user_A2MXXMQL8L7XMH] [user_A1GOVZZW9BE0YV] [user_A2LV8GKCUOXUS3] [user_A290DMYJAKNK2A] [user_AFHVXY2F2I7NQ] .'
    
    # 分析样本
    stats = analyze_sample(sample_text)
    
    # 打印分析结果
    print_analysis(stats)

if __name__ == "__main__":
    main() 