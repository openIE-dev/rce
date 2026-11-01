// rce emit verilog (spec 10.5, spec/proposals/emit-verilog.md): the step function
// of this program as one synthesizable Verilog-2005 module. One event per clock:
// when ev_valid is high at a rising edge of clk, the step on (ev_tag, ev_payload)
// is taken; out_* hold its outcome from that edge, and the state registers change
// only when out_applied is 1. rst (synchronous, active high) loads the initial state.
// rce-hdl/1 {"emit_count_width":1,"emit_slots":0,"emit_type_width":1,"emit_types":[],"emit_value_width":1,"error_width":1,"errors":[],"events":[{"fields":[],"name":"Up"},{"fields":[],"name":"Down"},{"fields":[],"name":"Left"},{"fields":[],"name":"Right"}],"examples":[{"events":[{"tag":2,"values":[]},{"tag":2,"values":[]},{"tag":0,"values":[]}],"line":2},{"events":[{"tag":0,"values":[]},{"tag":0,"values":[]},{"tag":0,"values":[]},{"tag":0,"values":[]}],"line":3}],"format":"rce-hdl/1","kinds":{},"module":"rce_Game","payload_width":1,"regions":[{"fields":[{"name":"x","offset":0,"type":"u8","width":8},{"name":"y","offset":8,"type":"u8","width":8},{"name":"won","offset":16,"type":"bool","width":1}],"name":"Game"}],"state_width":17,"tag_width":2,"trap_value_width":10,"trap_width":1,"traps":[{"kind":"overflow","type":"u8"}]}
module rce_Game (
    input  wire clk,
    input  wire rst,
    input  wire ev_valid,
    input  wire [1:0] ev_tag,
    input  wire [0:0] ev_payload,
    output reg  out_valid,
    output reg  out_applied,
    output reg  [0:0] out_error,
    output reg  [0:0] out_trap,
    output reg  [9:0] out_trap_value,
    output reg  [0:0] out_emit_count,
    output reg  [0:0] out_emit_type,
    output reg  [0:0] out_emit_value,
    output wire [16:0] state
);
    // the state registers, first field at bit 0 of `state`
    reg [7:0] s0_0; // Game.x: u8
    reg [7:0] nx0_0;
    reg [7:0] s0_1; // Game.y: u8
    reg [7:0] nx0_1;
    reg [0:0] s0_2; // Game.won: bool
    reg [0:0] nx0_2;
    assign state = {s0_2, s0_1, s0_0};
    // the step's working registers
    reg [0:0] k; // kept: the event does not apply
    reg [0:0] kerr; // the declared error of a `fail`, plus one
    reg [0:0] ktrap; // the trap, plus one
    reg signed [9:0] kval; // the trap's value
    reg [0:0] ecnt; // values emitted
    reg [0:0] sel0; // a step of Game was taken
    reg signed [8:0] l6; // x
    reg signed [8:0] l7; // y
    reg [0:0] l8; // won
    reg signed [8:0] l9; // held value
    reg signed [8:0] l10; // held value
    reg [0:0] r11; // returned from up
    reg signed [8:0] l12; // result of up
    reg signed [8:0] l13; // y
    reg signed [9:0] t14;
    reg [0:0] r15; // returned from wall
    reg [0:0] l16; // result of wall
    reg signed [8:0] l17; // x
    reg signed [1:0] l18; // y
    reg [0:0] q19; // and
    reg [0:0] l20; // east
    reg [0:0] q21; // and
    reg [0:0] l22; // south
    reg [0:0] q23; // and
    reg [0:0] l24; // north
    reg [0:0] q25; // and
    reg [0:0] l26; // west
    reg [0:0] q27; // or
    reg [0:0] q28; // or
    reg [0:0] q29; // or
    reg signed [8:0] l30; // held value
    reg [0:0] r31; // returned from up
    reg signed [8:0] l32; // result of up
    reg signed [8:0] l33; // y
    reg signed [9:0] t34;
    reg [0:0] q35; // or
    reg signed [8:0] l36; // held value
    reg [0:0] r37; // returned from on_flag
    reg [0:0] l38; // result of on_flag
    reg signed [8:0] l39; // x
    reg signed [1:0] l40; // y
    reg [0:0] q41; // and
    reg signed [8:0] l42; // x
    reg signed [8:0] l43; // y
    reg [0:0] l44; // won
    reg signed [8:0] l45; // held value
    reg signed [8:0] l46; // held value
    reg [0:0] r47; // returned from down
    reg signed [8:0] l48; // result of down
    reg signed [8:0] l49; // y
    reg signed [9:0] t50;
    reg signed [9:0] t51;
    reg [0:0] r52; // returned from wall
    reg [0:0] l53; // result of wall
    reg signed [8:0] l54; // x
    reg signed [1:0] l55; // y
    reg [0:0] q56; // and
    reg [0:0] l57; // east
    reg [0:0] q58; // and
    reg [0:0] l59; // south
    reg [0:0] q60; // and
    reg [0:0] l61; // north
    reg [0:0] q62; // and
    reg [0:0] l63; // west
    reg [0:0] q64; // or
    reg [0:0] q65; // or
    reg [0:0] q66; // or
    reg signed [8:0] l67; // held value
    reg [0:0] r68; // returned from down
    reg signed [8:0] l69; // result of down
    reg signed [8:0] l70; // y
    reg signed [9:0] t71;
    reg signed [9:0] t72;
    reg [0:0] q73; // or
    reg signed [8:0] l74; // held value
    reg [0:0] r75; // returned from on_flag
    reg [0:0] l76; // result of on_flag
    reg signed [8:0] l77; // x
    reg signed [1:0] l78; // y
    reg [0:0] q79; // and
    reg signed [8:0] l80; // x
    reg signed [8:0] l81; // y
    reg [0:0] l82; // won
    reg signed [8:0] l83; // held value
    reg [0:0] r84; // returned from left
    reg signed [8:0] l85; // result of left
    reg signed [8:0] l86; // x
    reg signed [9:0] t87;
    reg signed [8:0] l88; // held value
    reg [0:0] r89; // returned from wall
    reg [0:0] l90; // result of wall
    reg signed [1:0] l91; // x
    reg signed [8:0] l92; // y
    reg [0:0] l93; // east
    reg [0:0] l94; // south
    reg [0:0] l95; // north
    reg [0:0] l96; // west
    reg [0:0] q97; // or
    reg [0:0] q98; // or
    reg signed [8:0] l99; // held value
    reg [0:0] r100; // returned from left
    reg signed [8:0] l101; // result of left
    reg signed [8:0] l102; // x
    reg signed [9:0] t103;
    reg [0:0] q104; // or
    reg signed [8:0] l105; // held value
    reg [0:0] r106; // returned from on_flag
    reg [0:0] l107; // result of on_flag
    reg signed [1:0] l108; // x
    reg signed [8:0] l109; // y
    reg signed [8:0] l110; // x
    reg signed [8:0] l111; // y
    reg [0:0] l112; // won
    reg signed [8:0] l113; // held value
    reg [0:0] r114; // returned from right
    reg signed [8:0] l115; // result of right
    reg signed [8:0] l116; // x
    reg signed [9:0] t117;
    reg signed [9:0] t118;
    reg signed [8:0] l119; // held value
    reg [0:0] r120; // returned from wall
    reg [0:0] l121; // result of wall
    reg signed [1:0] l122; // x
    reg signed [8:0] l123; // y
    reg [0:0] l124; // east
    reg [0:0] l125; // south
    reg [0:0] l126; // north
    reg [0:0] l127; // west
    reg [0:0] q128; // or
    reg [0:0] q129; // or
    reg signed [8:0] l130; // held value
    reg [0:0] r131; // returned from right
    reg signed [8:0] l132; // result of right
    reg signed [8:0] l133; // x
    reg signed [9:0] t134;
    reg signed [9:0] t135;
    reg [0:0] q136; // or
    reg signed [8:0] l137; // held value
    reg [0:0] r138; // returned from on_flag
    reg [0:0] l139; // result of on_flag
    reg signed [1:0] l140; // x
    reg signed [8:0] l141; // y
    always @* begin
        nx0_0 = s0_0;
        nx0_1 = s0_1;
        nx0_2 = s0_2;
        k = 1'd0;
        kerr = 1'd0;
        ktrap = 1'd0;
        kval = 10'd0;
        ecnt = 1'd0;
        sel0 = 1'd0;
        l6 = 9'd0;
        l7 = 9'd0;
        l8 = 1'd0;
        l9 = 9'd0;
        l10 = 9'd0;
        r11 = 1'd0;
        l12 = 9'd0;
        l13 = 9'd0;
        t14 = 10'd0;
        r15 = 1'd0;
        l16 = 1'd0;
        l17 = 9'd0;
        l18 = 2'd0;
        q19 = 1'd0;
        l20 = 1'd0;
        q21 = 1'd0;
        l22 = 1'd0;
        q23 = 1'd0;
        l24 = 1'd0;
        q25 = 1'd0;
        l26 = 1'd0;
        q27 = 1'd0;
        q28 = 1'd0;
        q29 = 1'd0;
        l30 = 9'd0;
        r31 = 1'd0;
        l32 = 9'd0;
        l33 = 9'd0;
        t34 = 10'd0;
        q35 = 1'd0;
        l36 = 9'd0;
        r37 = 1'd0;
        l38 = 1'd0;
        l39 = 9'd0;
        l40 = 2'd0;
        q41 = 1'd0;
        l42 = 9'd0;
        l43 = 9'd0;
        l44 = 1'd0;
        l45 = 9'd0;
        l46 = 9'd0;
        r47 = 1'd0;
        l48 = 9'd0;
        l49 = 9'd0;
        t50 = 10'd0;
        t51 = 10'd0;
        r52 = 1'd0;
        l53 = 1'd0;
        l54 = 9'd0;
        l55 = 2'd0;
        q56 = 1'd0;
        l57 = 1'd0;
        q58 = 1'd0;
        l59 = 1'd0;
        q60 = 1'd0;
        l61 = 1'd0;
        q62 = 1'd0;
        l63 = 1'd0;
        q64 = 1'd0;
        q65 = 1'd0;
        q66 = 1'd0;
        l67 = 9'd0;
        r68 = 1'd0;
        l69 = 9'd0;
        l70 = 9'd0;
        t71 = 10'd0;
        t72 = 10'd0;
        q73 = 1'd0;
        l74 = 9'd0;
        r75 = 1'd0;
        l76 = 1'd0;
        l77 = 9'd0;
        l78 = 2'd0;
        q79 = 1'd0;
        l80 = 9'd0;
        l81 = 9'd0;
        l82 = 1'd0;
        l83 = 9'd0;
        r84 = 1'd0;
        l85 = 9'd0;
        l86 = 9'd0;
        t87 = 10'd0;
        l88 = 9'd0;
        r89 = 1'd0;
        l90 = 1'd0;
        l91 = 2'd0;
        l92 = 9'd0;
        l93 = 1'd0;
        l94 = 1'd0;
        l95 = 1'd0;
        l96 = 1'd0;
        q97 = 1'd0;
        q98 = 1'd0;
        l99 = 9'd0;
        r100 = 1'd0;
        l101 = 9'd0;
        l102 = 9'd0;
        t103 = 10'd0;
        q104 = 1'd0;
        l105 = 9'd0;
        r106 = 1'd0;
        l107 = 1'd0;
        l108 = 2'd0;
        l109 = 9'd0;
        l110 = 9'd0;
        l111 = 9'd0;
        l112 = 1'd0;
        l113 = 9'd0;
        r114 = 1'd0;
        l115 = 9'd0;
        l116 = 9'd0;
        t117 = 10'd0;
        t118 = 10'd0;
        l119 = 9'd0;
        r120 = 1'd0;
        l121 = 1'd0;
        l122 = 2'd0;
        l123 = 9'd0;
        l124 = 1'd0;
        l125 = 1'd0;
        l126 = 1'd0;
        l127 = 1'd0;
        q128 = 1'd0;
        q129 = 1'd0;
        l130 = 9'd0;
        r131 = 1'd0;
        l132 = 9'd0;
        l133 = 9'd0;
        t134 = 10'd0;
        t135 = 10'd0;
        q136 = 1'd0;
        l137 = 9'd0;
        r138 = 1'd0;
        l139 = 1'd0;
        l140 = 2'd0;
        l141 = 9'd0;
        // step Game on Up (line 59)
        if (!k && !sel0) begin
            l6 = $signed({1'b0, s0_0[7:0]});
            l7 = $signed({1'b0, s0_1[7:0]});
            l8 = s0_2[0];
            if ((ev_tag == 2'd0)) begin
                l9 = l6;
                l10 = l7;
                r11 = 1'b0;
                l12 = 2'sd0;
                l13 = l10;
                if ((l13 == 2'sd0)) begin
                    l12 = 4'sd7;
                    r11 = 1'b1;
                end
                if (!k && !r11) begin
                    t14 = l13 - 2'sd1;
                    if (!k && (t14 < 2'sd0 || t14 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t14; end
                    l12 = t14;
                    r11 = 1'b1;
                end
                r15 = 1'b0;
                l16 = 1'b0;
                l17 = l9;
                l18 = l12;
                q19 = (l17 == 3'sd3);
                if (q19) begin
                    q19 = 1'b1;
                end
                l20 = q19;
                q21 = (l17 <= 3'sd3);
                if (q21) begin
                    q21 = 1'b0;
                end
                l22 = q21;
                q23 = (l17 <= 3'sd3);
                if (q23) begin
                    q23 = 1'b0;
                end
                l24 = q23;
                q25 = (l17 == 5'sd11);
                if (q25) begin
                    q25 = 1'b1;
                end
                l26 = q25;
                q27 = l20;
                if (!q27) begin
                    q27 = l22;
                end
                q28 = q27;
                if (!q28) begin
                    q28 = l24;
                end
                q29 = q28;
                if (!q29) begin
                    q29 = l26;
                end
                l16 = q29;
                r15 = 1'b1;
                if (!k && (!l16)) begin
                    sel0 = 1'b1;
                    l30 = l7;
                    r31 = 1'b0;
                    l32 = 2'sd0;
                    l33 = l30;
                    if ((l33 == 2'sd0)) begin
                        l32 = 4'sd7;
                        r31 = 1'b1;
                    end
                    if (!k && !r31) begin
                        t34 = l33 - 2'sd1;
                        if (!k && (t34 < 2'sd0 || t34 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t34; end
                        l32 = t34;
                        r31 = 1'b1;
                    end
                    l7 = l32;
                    q35 = l8;
                    if (!q35) begin
                        l36 = l6;
                        r37 = 1'b0;
                        l38 = 1'b0;
                        l39 = l36;
                        l40 = l7;
                        q41 = (l39 == 2'sd1);
                        if (q41) begin
                            q41 = 1'b0;
                        end
                        l38 = q41;
                        r37 = 1'b1;
                        q35 = l38;
                    end
                    l8 = q35;
                    nx0_0[7:0] = l6;
                    nx0_1[7:0] = l7;
                    nx0_2[0] = l8;
                end
            end
        end
        // step Game on Down (line 62)
        if (!k && !sel0) begin
            l42 = $signed({1'b0, s0_0[7:0]});
            l43 = $signed({1'b0, s0_1[7:0]});
            l44 = s0_2[0];
            if ((ev_tag == 2'd1)) begin
                l45 = l42;
                l46 = l43;
                r47 = 1'b0;
                l48 = 2'sd0;
                l49 = l46;
                t50 = l49 + 2'sd1;
                if (!k && (t50 < 2'sd0 || t50 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t50; end
                if ((t50 == 5'sd8)) begin
                    l48 = 2'sd0;
                    r47 = 1'b1;
                end
                if (!k && !r47) begin
                    t51 = l49 + 2'sd1;
                    if (!k && (t51 < 2'sd0 || t51 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t51; end
                    l48 = t51;
                    r47 = 1'b1;
                end
                r52 = 1'b0;
                l53 = 1'b0;
                l54 = l45;
                l55 = l48;
                q56 = (l54 == 3'sd3);
                if (q56) begin
                    q56 = 1'b1;
                end
                l57 = q56;
                q58 = (l54 <= 3'sd3);
                if (q58) begin
                    q58 = 1'b0;
                end
                l59 = q58;
                q60 = (l54 <= 3'sd3);
                if (q60) begin
                    q60 = 1'b0;
                end
                l61 = q60;
                q62 = (l54 == 5'sd11);
                if (q62) begin
                    q62 = 1'b1;
                end
                l63 = q62;
                q64 = l57;
                if (!q64) begin
                    q64 = l59;
                end
                q65 = q64;
                if (!q65) begin
                    q65 = l61;
                end
                q66 = q65;
                if (!q66) begin
                    q66 = l63;
                end
                l53 = q66;
                r52 = 1'b1;
                if (!k && (!l53)) begin
                    sel0 = 1'b1;
                    l67 = l43;
                    r68 = 1'b0;
                    l69 = 2'sd0;
                    l70 = l67;
                    t71 = l70 + 2'sd1;
                    if (!k && (t71 < 2'sd0 || t71 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t71; end
                    if ((t71 == 5'sd8)) begin
                        l69 = 2'sd0;
                        r68 = 1'b1;
                    end
                    if (!k && !r68) begin
                        t72 = l70 + 2'sd1;
                        if (!k && (t72 < 2'sd0 || t72 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t72; end
                        l69 = t72;
                        r68 = 1'b1;
                    end
                    l43 = l69;
                    q73 = l44;
                    if (!q73) begin
                        l74 = l42;
                        r75 = 1'b0;
                        l76 = 1'b0;
                        l77 = l74;
                        l78 = l43;
                        q79 = (l77 == 2'sd1);
                        if (q79) begin
                            q79 = 1'b0;
                        end
                        l76 = q79;
                        r75 = 1'b1;
                        q73 = l76;
                    end
                    l44 = q73;
                    nx0_0[7:0] = l42;
                    nx0_1[7:0] = l43;
                    nx0_2[0] = l44;
                end
            end
        end
        // step Game on Left (line 65)
        if (!k && !sel0) begin
            l80 = $signed({1'b0, s0_0[7:0]});
            l81 = $signed({1'b0, s0_1[7:0]});
            l82 = s0_2[0];
            if ((ev_tag == 2'd2)) begin
                l83 = l80;
                r84 = 1'b0;
                l85 = 2'sd0;
                l86 = l83;
                if ((l86 == 2'sd0)) begin
                    l85 = 5'sd11;
                    r84 = 1'b1;
                end
                if (!k && !r84) begin
                    t87 = l86 - 2'sd1;
                    if (!k && (t87 < 2'sd0 || t87 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t87; end
                    l85 = t87;
                    r84 = 1'b1;
                end
                l88 = l81;
                r89 = 1'b0;
                l90 = 1'b0;
                l91 = l85;
                l92 = l88;
                l93 = 1'b0;
                l94 = (l92 == 3'sd3);
                l95 = (l92 == 4'sd7);
                l96 = 1'b0;
                q97 = l94;
                if (!q97) begin
                    q97 = l95;
                end
                q98 = q97;
                if (!q98) begin
                    q98 = l96;
                end
                l90 = q98;
                r89 = 1'b1;
                if (!k && (!l90)) begin
                    sel0 = 1'b1;
                    l99 = l80;
                    r100 = 1'b0;
                    l101 = 2'sd0;
                    l102 = l99;
                    if ((l102 == 2'sd0)) begin
                        l101 = 5'sd11;
                        r100 = 1'b1;
                    end
                    if (!k && !r100) begin
                        t103 = l102 - 2'sd1;
                        if (!k && (t103 < 2'sd0 || t103 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t103; end
                        l101 = t103;
                        r100 = 1'b1;
                    end
                    l80 = l101;
                    q104 = l82;
                    if (!q104) begin
                        l105 = l81;
                        r106 = 1'b0;
                        l107 = 1'b0;
                        l108 = l80;
                        l109 = l105;
                        l107 = 1'b0;
                        r106 = 1'b1;
                        q104 = l107;
                    end
                    l82 = q104;
                    nx0_0[7:0] = l80;
                    nx0_1[7:0] = l81;
                    nx0_2[0] = l82;
                end
            end
        end
        // step Game on Right (line 68)
        if (!k && !sel0) begin
            l110 = $signed({1'b0, s0_0[7:0]});
            l111 = $signed({1'b0, s0_1[7:0]});
            l112 = s0_2[0];
            if ((ev_tag == 2'd3)) begin
                l113 = l110;
                r114 = 1'b0;
                l115 = 2'sd0;
                l116 = l113;
                t117 = l116 + 2'sd1;
                if (!k && (t117 < 2'sd0 || t117 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t117; end
                if ((t117 == 5'sd12)) begin
                    l115 = 2'sd0;
                    r114 = 1'b1;
                end
                if (!k && !r114) begin
                    t118 = l116 + 2'sd1;
                    if (!k && (t118 < 2'sd0 || t118 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t118; end
                    l115 = t118;
                    r114 = 1'b1;
                end
                l119 = l111;
                r120 = 1'b0;
                l121 = 1'b0;
                l122 = l115;
                l123 = l119;
                l124 = 1'b0;
                l125 = (l123 == 3'sd3);
                l126 = (l123 == 4'sd7);
                l127 = 1'b0;
                q128 = l125;
                if (!q128) begin
                    q128 = l126;
                end
                q129 = q128;
                if (!q129) begin
                    q129 = l127;
                end
                l121 = q129;
                r120 = 1'b1;
                if (!k && (!l121)) begin
                    sel0 = 1'b1;
                    l130 = l110;
                    r131 = 1'b0;
                    l132 = 2'sd0;
                    l133 = l130;
                    t134 = l133 + 2'sd1;
                    if (!k && (t134 < 2'sd0 || t134 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t134; end
                    if ((t134 == 5'sd12)) begin
                        l132 = 2'sd0;
                        r131 = 1'b1;
                    end
                    if (!k && !r131) begin
                        t135 = l133 + 2'sd1;
                        if (!k && (t135 < 2'sd0 || t135 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t135; end
                        l132 = t135;
                        r131 = 1'b1;
                    end
                    l110 = l132;
                    q136 = l112;
                    if (!q136) begin
                        l137 = l111;
                        r138 = 1'b0;
                        l139 = 1'b0;
                        l140 = l110;
                        l141 = l137;
                        l139 = 1'b0;
                        r138 = 1'b1;
                        q136 = l139;
                    end
                    l112 = q136;
                    nx0_0[7:0] = l110;
                    nx0_1[7:0] = l111;
                    nx0_2[0] = l112;
                end
            end
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            s0_0 <= 8'h08;
            s0_1 <= 8'h05;
            s0_2 <= 1'h0;
            out_valid <= 1'd0;
            out_applied <= 1'd0;
            out_error <= 1'd0;
            out_trap <= 1'd0;
            out_trap_value <= 10'd0;
            out_emit_count <= 1'd0;
            out_emit_type <= 1'd0;
            out_emit_value <= 1'd0;
        end else begin
            out_valid <= ev_valid;
            if (ev_valid) begin
                out_applied <= !k;
                out_error <= kerr;
                out_trap <= ktrap;
                out_trap_value <= kval;
                out_emit_count <= k ? 1'd0 : ecnt;
                out_emit_type <= 1'd0;
                out_emit_value <= 1'd0;
                if (!k) begin
                    s0_0 <= nx0_0;
                    s0_1 <= nx0_1;
                    s0_2 <= nx0_2;
                end
            end
        end
    end
endmodule
