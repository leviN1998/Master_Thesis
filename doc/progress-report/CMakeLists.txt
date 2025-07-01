cmake_minimum_required(VERSION 3.9)

# Use the commented-out lines to use the following  folder structure
# progressreport/
#                ieee.csl
#                pandocology.cmake
#                subtitle.tex
#                1/CMakeLists.txt
#                1/progressreport.md
#                1/progressreport.bib
#                1/build/build
#                2/MakeLists.txt
#                2/progressreport.md
#                2/progressreport.bib
#                2/build/build
#                ...
#LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")
INCLUDE(pandocology)

add_document(
    TARGET progressreport
    OUTPUT_FILE progressreport.pdf
    SOURCES progressreport.md
    #RESOURCE_FILES progressreport.bib ../subtitle.tex ../ieee.csl
    RESOURCE_FILES progressreport.bib ./subtitle.tex ./ieee.csl
    PANDOC_DIRECTIVES --citeproc
                      --bibliography progressreport.bib
                      --csl          ieee.csl
                      -H             subtitle.tex
)
