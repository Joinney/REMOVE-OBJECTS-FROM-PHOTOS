
const guessJsonIndent=(jsonString)=>{
const firstIndex=skipWhitespaces(jsonString,0);

if(
firstIndex===undefined||
!isJsonObjectOrArray(jsonString[firstIndex]))
{
return
}

const secondIndex=skipWhitespaces(jsonString,firstIndex+1);

if(secondIndex===undefined){
return
}

return getIndent(jsonString,firstIndex,secondIndex)
};

export default guessJsonIndent;




const skipWhitespaces=(jsonString,startIndex)=>{
for(let index=startIndex;index<jsonString.length;index+=1){
const character=jsonString[index];

if(!isJsonWhitespace(character)){
return index
}
}
};



const isJsonWhitespace=(character)=>
character===" "||
character==="\t"||
character==="\n"||
character==="\r";



const isJsonObjectOrArray=(character)=>
character==="{"||character==="[";




const getIndent=(jsonString,firstIndex,secondIndex)=>{
let indent;

for(let index=secondIndex-1;index>firstIndex;index-=1){
const character=jsonString[index];

if(character==="\r"){
return
}

if(character==="\n"){
return normalizeIndent(indent)
}

if(indent===undefined){
indent=character
}else if(indent[0]===character){
indent+=character
}else{
return
}
}
};




const normalizeIndent=(indent)=>{
if(indent===undefined){
return 0
}

return indent[0]===" "?indent.length:indent
};