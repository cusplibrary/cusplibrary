/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file convergence_stub.h
 *  \brief Javascript convergence html stub
 */

namespace cusp
{
namespace plot
{
namespace detail
{

const char * convergence_stub =

"<!doctype>"
"<script type='text/javascript' src='http://cdnjs.cloudflare.com/ajax/libs/d3/3.4.12/d3.min.js'></script>"
"<script type='text/javascript' src='http://cdnjs.cloudflare.com/ajax/libs/rickshaw/1.4.6/rickshaw.min.js'></script>"
"<link   type='text/css' rel='stylesheet' href='http://cdnjs.cloudflare.com/ajax/libs/rickshaw/1.4.6/rickshaw.min.css'>"
"<style>"
"#chart_container {"
"	float: left;"
"	position: relative;"
"}"
"#chart {"
"  position: relative;"
"  left: 60px;"
"}"
"#x_axis {"
"  position: absolute;"
"  left : 60px;"
"}"
"#y_axis {"
"  position: absolute;"
"  top: 0;"
"  bottom: 0px;"
"  width: 60px;"
"}"
"#legend {"
"  position: relative;"
"  text-align: center;"
"  top: 40;"
"}"
".swatch {"
"      display: inline-block;"
"      width: 10px;"
"      height: 10px;"
"      margin: 0 8px 0 0;"
"}"
".label {"
"  display: inline-block;"
"}"
".line {"
"     display: inline-block;"
"     margin: 0 0 0 30px;"
"}"
"div, span, p, td {"
"	font-family: Arial, sans-serif;"
"}"
"</style>"
"<div id='chart_container'>"
"  <div id='chart'></div>"
"  <div id='x_axis'></div>"
"  <div id='y_axis'></div>"
"  <div id='legend'></div>"
"</div>"
"<script>"
"//data"
"var data = [];"
"var datacount = [];"
"var max_len = 0;"
"var min = Number.MAX_VALUE;"
"var max = Number.MIN_VALUE;"
"var palette = new Rickshaw.Color.Palette();"
"for(i = 0; i < res.length; i++) {"
"  datacount[i] = res[i].length;"
"  max_len = Math.max(max_len, datacount[i]);"
"  var points = d3.range(datacount[i]).map(function(j) {"
"                y = res[i][j];"
"                min = Math.min(min, y);"
"                max = Math.max(max, y);"
"                return { x: j, y: y };"
"             }, self);"
"  data[i] = {"
"              name : names[i],"
"              color: palette.color(),"
"              data : points,"
"            };"
"}"
"var logScale = d3.scale.log().domain([min, max]);"
"for(i = 0; i < res.length; i++) {"
"  data[i].scale = logScale;"
"}"
"var graph = new Rickshaw.Graph( {"
"        element: document.querySelector('#chart'),"
"        width: 800,"
"        height: 800,"
"        renderer: 'line',"
"        series: Array.apply(null, data)"
"} );"
"var num_xticks = Math.min(20, max_len);"
"var y_tick_values = d3.range(max_len).map(function(i) {"
"                return max / Math.pow(10,i);"
"                }, self);"
"formatSCI = function(y) {"
"  var abs_y = Math.abs(y);"
"  return y.toExponential(1)"
"};"
"var x_ticks = new Rickshaw.Graph.Axis.X( {"
"  graph         : graph,"
"  orientation   : 'bottom',"
"  element       : document.getElementById('x_axis'),"
"  ticks         : num_xticks,"
"  tickFormat    : Rickshaw.Fixtures.Number.formatKMBT,"
"  pixelsPerTick : 200,"
"} );"
"var y_ticks = new Rickshaw.Graph.Axis.Y.Scaled( {"
"  graph       : graph,"
"  orientation : 'left',"
"  element     : document.getElementById('y_axis'),"
"  tickFormat  : formatSCI,"
"  tickValues  : y_tick_values,"
"  scale       : logScale"
"} );"
"var legend = document.querySelector('#legend');"
"graph.render();"
"var Hover = Rickshaw.Class.create(Rickshaw.Graph.HoverDetail, {"
"  render: function(args) {"
"    legend.innerHTML = 'Iteration: ' + args.domainX;"
"    args.detail.sort(function(a, b) { return a.order - b.order }).forEach( function(d) {"
"      var line = document.createElement('div');"
"      line.className = 'line';"
"      var swatch = document.createElement('div');"
"      swatch.className = 'swatch';"
"      swatch.style.backgroundColor = d.series.color;"
"      var label = document.createElement('div');"
"      label.className = 'label';"
"      label.innerHTML = d.name + ': ' + d.formattedYValue;"
"      line.appendChild(swatch);"
"      line.appendChild(label);"
"      legend.appendChild(line);"
"      var dot = document.createElement('div');"
"      dot.className = 'dot';"
"      dot.style.top = graph.y(d.value.y0 + d.value.y) + 'px';"
"      dot.style.borderColor = d.series.color;"
"      this.element.appendChild(dot);"
"      dot.className = 'dot active';"
"      this.show();"
"    }, this );"
"        }"
"});"
"var hover = new Hover( { graph: graph } );"
"</script>";

} // end namespace detail
} // end namespace plot
} // end namespace cusp
