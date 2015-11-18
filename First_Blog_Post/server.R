library(choroplethr)
library(choroplethrMaps)
library(ggplot2)
library(shiny)
data("county.regions")

ObesityData = readRDS("data/ObesityData")
IncomeData = readRDS("data/IncomeData")
MortalityData = readRDS("data/MortalityData")
DiagnosedData = readRDS("data/DiagnosedData")
new_york_city = c(36005, 36047, 36061, 36081, 36085)

shinyServer(
  function(input, output) {
    output$map = renderPlot({
      data <- switch(input$var1,
                     "Income" = IncomeData,
                     "Obesity Rate"= ObesityData,
                     "Diabetic Mortality" = MortalityData,
                     "Diabetic Rate" = DiagnosedData)
      
      mapcolors <- switch(input$var1,
                          "Income" = scale_fill_brewer(palette=2),
                          "Obesity Rate"= scale_fill_brewer(palette=3),
                          "Diabetic Mortality" = scale_fill_brewer(palette=4),
                          "Diabetic Rate" = scale_fill_brewer(palette=5))
      
      if (input$radio == "Just New York City"){
        county_choropleth(data,
                        title       =  input$radio,
                        legend      = input$var1,
                        num_colors  = 9,
                        county_zoom = new_york_city) + mapcolors}
      else{
        county_choropleth(data,
                          title       =  input$radio,
                          legend      = input$var1,
                          num_colors  = 9,
                          state_zoom  = c("new york")) + mapcolors}
      

                        
    })
    output$maptext <- renderText({
      if (input$radio == "Just New York City" & input$var1 == "Income" ){
        "Notice the disparity between Manhattan and the Bronx"}
      else if (input$radio == "All New York State" & input$var1 == "Income" ){
        "Notice the wealthiest counties are concentrated in the South"}
      else if (input$radio == "Just New York City" & input$var1 == "Obesity Rate"){
        "Notice that the obesity rates are the highest,
yet their income and diabetic rates are very different
        "}
      else if (input$radio == "All New York State" & input$var1 == "Obesity Rate" ){
        "Notice as you move North and West, the obesity rate increases"}
      else if (input$radio == "Just New York City" & input$var1 == "Diabetic Mortality"){
        "The mortality rates mirror the diabetic rates, the Bronx has the highest diabetic mortality rate"}
      else if (input$radio == "All New York State" & input$var1 == "Diabetic Mortality" ){
        "The diabetic mortality rate mirrors the diabetic rate.  It should be noted that this mortality
        rate reflects the percentage of all deaths resulting from diabetic complications"}
      else if (input$radio == "Just New York City" & input$var1 == "Diabetic Rate"){
        "The Bronx has the highest diabetic rate but is closely followed by Queens and Kings County"}
      else if (input$radio == "All New York State" & input$var1 == "Diabetic Rate"){
        "As you move away from NYC the diabetic rate goes up"}
      else {"Yes this is text"}
      
      
    })
    
    
    output$plot = renderPlot({
      datax <- switch(input$xvar,
                     "Income" = IncomeData$value,
                     "Obesity Rate"= ObesityData$value,
                     "Diabetic Mortality" = MortalityData$value,
                     "Diabetic Rate" = DiagnosedData$value)
      datay<- switch(input$yvar,
                     "Income" = IncomeData$value,
                     "Obesity Rate"= ObesityData$value,
                     "Diabetic Mortality" = MortalityData$value,
                     "Diabetic Rate" = DiagnosedData$value)
      
      title = paste0(input$xvar," vs ", input$yvar," in New York Counties")
      qplot(datax, datay, xlab = input$xvar, ylab = input$yvar,
            main = title ) + (geom_smooth())
      
      
    })
  }
)

