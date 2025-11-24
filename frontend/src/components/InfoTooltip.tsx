import React, { useState } from "react";
import "./InfoTooltip.css";

interface InfoTooltipProps {
  text: string;
}

const InfoTooltip: React.FC<InfoTooltipProps> = ({ text }) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <span
      className="info-tooltip-container"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      <span className="info-icon">ⓘ</span>
      {isVisible && <div className="tooltip-bubble">{text}</div>}
    </span>
  );
};

export default InfoTooltip;
