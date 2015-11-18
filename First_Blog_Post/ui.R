library(shiny)
library(ggvis)

shinyUI(fluidPage(
  
  titlePanel("New York State Diabetic Rates"),
  
  tabsetPanel(
      tabPanel("map",        
               sidebarLayout(
                 sidebarPanel(      
                   
                   selectInput("var1",
                               label = "Select Data Set",
                               choices = c("Income","Diabetic Mortality","Diabetic Rate","Obesity Rate"),
                               selected = "Diabetic Rate"),
                   
                   radioButtons("radio", label = h3("Counties"),
                                choices = c("All New York State", "Just New York City"), 
                                selected = "Just New York City"),
                   img(src = "new-york-county-map-90pct.GIF", height = 250, width = 250)
                   
                   
                   #hr(),
                   #fluidRow(column(3, verbatimTextOutput("value")))
                 ),
                 mainPanel(
                   plotOutput("map"),
                   textOutput("maptext")
                   
                 )
               )
               ),
      tabPanel("plot",
               sidebarLayout(
                 sidebarPanel(
                   selectInput("xvar",
                               label = "Select X Variable",
                               choices = c("Income","Diabetic Mortality","Diabetic Rate","Obesity Rate"),
                               selected = "Diabetic Rate"),
                   selectInput("yvar",
                               label = "Select Y Variable",
                               choices = c("Income","Diabetic Mortality","Diabetic Rate","Obesity Rate"),
                               selected = "Diabetic Rate")
                   
                   
                               ),
                 
               
               mainPanel(plotOutput("plot"))
               )
      
  )
  )
)
)