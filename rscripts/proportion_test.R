proptest <- function(success, counts){
  # print(success)
  # print(counts)
  prop = prop.test(success, counts)
  # print(prop)
  return(prop)
}