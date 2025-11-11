# 获取 Decoder_NEXYS_A7_100T 模块的单元名称
set decoder_cells [get_cells -hierarchical -filter {NAME =~ "*DEC*"}]

# 禁用 FSM 提取优化和其他优化
foreach cell $decoder_cells {
    # 禁用 FSM 提取
    set_property DONT_TOUCH true $cell
    set_property keep_hierarchy true $cell
    
    # 禁用组合逻辑优化
    set_property NO_LC true $cell
    set_property NO_SHREG_EXTRACT true $cell
}
